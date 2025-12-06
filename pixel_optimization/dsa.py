import torch
import numpy as np
import time

from distance import gradient_distance
from common import define_model
from pixel_optimization.augment import DiffAugment, ParamDiffAug
from pixel_optimization.pix_utils import get_images, get_loops, update_model, evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def condensation(
        args,
        step, 
        channel,
        img_size,
        outer_loop, 
        inner_loop, 
        eval_model,
        net_params, 
        loss_fn, 
        optimizer_img, 
        optimizer_model,
        data_syn,
        target_syn,
        indices_class, 
        images_all): 
    
    loss_avg = 0.0
    for t in range(outer_loop):
        loss = torch.tensor(0.0).to(DEVICE)

        for c in range(args.num_classes):
            img_real = get_images(indices_class, images_all, c, args.batch_size)
            img_syn = data_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, img_size[0], img_size[1]))

            img_real = DiffAugment(img_real, strategy=args.aug_type, param=ParamDiffAug())
            img_syn = DiffAugment(img_syn, strategy=args.aug_type, param=ParamDiffAug())

            target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=DEVICE) * c
            prediction_real = eval_model(img_real)
            loss_real = loss_fn(prediction_real, target_real)
            gw_real = torch.autograd.grad(loss_real, net_params)
            gw_real = list((_.detach().clone() for _ in gw_real))
            
            target_syn_tmp = torch.ones((args.ipc,), dtype=torch.long, device=DEVICE) * c
            prediction_syn = eval_model(img_syn)
            loss_syn = loss_fn(prediction_syn, target_syn_tmp)
            gw_syn = torch.autograd.grad(loss_syn, net_params, create_graph=True)

            dist = gradient_distance(gw_syn, gw_real, DEVICE)
            loss += dist

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()
        if t == args.ipc - 1: 
            break
        update_model(optimizer_model, inner_loop, loss_fn, eval_model, data_syn, target_syn)

    loss_avg /= (args.num_classes * args.ipc) 
    if step % args.print_freq == 0: 
        print(f"Step {step + 1}, Loss: {loss_avg:.4f}")
        model_save_name = f'{args.eval_model[0]}_ipc{args.ipc}_step{step}.pt'
        path = f'{args.output_dir}/outputs/{model_save_name}'

        torch.save(data_syn, path)
    pass

def start_dsa(args, trainset, testset): 
    ipc = args.ipc 

    outer_loop, inner_loop = get_loops(ipc)

    if args.data == 'fmnist' or args.data == 'mnist':
        img_size = (28, 28)
        channel = 1
    else: 
        img_size = (32, 32)
        channel = 3

    indices_class = [[] for c in range(args.num_classes)] 
    images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
    labels_all = [trainset[i][1] for i in range(len(trainset))]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    images_all = torch.cat(images_all, dim=0).to(DEVICE)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=DEVICE)


    data_syn = torch.randn(size=(args.num_classes * args.ipc, channel, img_size[0], img_size[1]), dtype=torch.float, requires_grad=True, device=DEVICE)
    targets_syn = torch.tensor([np.ones(args.ipc)*i for i in range(args.num_classes)], dtype=torch.long, requires_grad=False,  device=DEVICE).view(-1)
    
    optimizer_img = torch.optim.SGD([data_syn, ], lr=args.lr_img)
    optimizer_img.zero_grad()

    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE) 

    start_time = time.time()

    for step in range(args.epochs):
        eval_model = define_model(args, args.num_classes, 'single').to(DEVICE)
        net_params = list(eval_model.parameters())

        optimizer_net = torch.optim.SGD(eval_model.parameters(), lr=args.lr)
        optimizer_net.zero_grad()
        condensation(
            args,
            step, 
            channel,
            img_size,
            outer_loop, 
            inner_loop, 
            eval_model,
            net_params, 
            loss_fn, 
            optimizer_img, 
            optimizer_net,
            data_syn,
            targets_syn,
            indices_class, 
            images_all)
 
    end_time = time.time()
    
    acc, elapsed_time = evaluate(
        args,
        [data_syn, ],
        torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        ),
        args.batch_size,
        args.ipc,
        args.epochs_eval,
        args.num_classes,
        DEVICE
    )
    print(f'Algorithm: DSA')
    print(f"Total condensation time: {end_time - start_time:.2f} seconds")
    print(f"Final evaluation accuracy: {acc:.2f} %, Time for training: {elapsed_time:.2f} seconds")

