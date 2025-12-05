import torch
import numpy as np
import time

from common import define_model
from pix_utils import get_images, evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def condensation(
        args,
        step, 
        channel,
        img_size,
        eval_model, 
        optimizer_img, 
        data_syn,
        indices_class, 
        images_all): 
    
    loss_avg = 0.0
    loss = torch.tensor(0.0).to(DEVICE)

    image_real_all, image_syn_all = [], []

    for c in range(args.num_classes):
        img_real = get_images(indices_class, images_all, c, args.batch_size)
        img_syn = data_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, img_size[0], img_size[1]))

        image_real_all.append(img_real)
        image_syn_all.append(img_syn)

    img_real_all = torch.cat(image_real_all, dim=0)
    image_syn_all = torch.cat(image_syn_all, dim=0)

    _, output_real = eval_model(img_real_all, return_features=True)
    _, output_syn = eval_model(image_syn_all, return_features=True)
    
    loss += torch.sum((torch.mean(output_real.reshape(args.num_classes, args.batch_size, -1), dim=1) - torch.mean(output_syn.reshape(args.num_classes, args.ipc, -1), dim=1))**2)


    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()
    loss_avg += loss.item()

    loss_avg /= args.num_classes

    if step % args.print_freq == 0: 
        print(f"Step {step + 1}, Loss: {loss_avg:.4f}")
        model_save_name = f'{args.eval_model[0]}_ipc{args.ipc}_step{step}.pt'
        path = f'{args.output_dir}/outputs/{model_save_name}'

        torch.save(data_syn, path)
    pass

def start_dm(args, trainset, testset): 
    ipc = args.ipc 

    if args.data == 'fmnist' or args.data == 'mnist':
        img_size = (28, 28)
        channel = 1
    else: 
        img_size = (32, 32)
        channel = 3

    indices_class = [[] for c in range(args.num_classese)] 
    images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
    labels_all = [trainset[i][1] for i in range(len(trainset))]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    images_all = torch.cat(images_all, dim=0).to(DEVICE)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=DEVICE)


    data_syn = torch.randn(size=(args.num_classes * args.ipc, channel, img_size[0], img_size[1]), dtype=torch.float, requires_grad=True, device=DEVICE)
    targets_syn = torch.tensor([np.ones(args.ipc)*i for i in range(args.num_classes)], dtype=torch.long, requires_grad=False,  device=DEVICE).view(-1)
    
    optimizer_img = torch.optim.SGD([data_syn, ], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    start_time = time.time()

    for step in range(args.epochs):
        eval_model = define_model(args, args.num_classes, 'single').to(DEVICE)
        net_params = list(eval_model.parameters())
        eval_model.train()

        for param in list(eval_model.parameters()):
            param.requires_grad = False

        condensation(
            args,
            step, 
            channel,
            img_size,
            eval_model,
            optimizer_img, 
            data_syn,
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
    print(f'Algorithm: DM')
    print(f"Total condensation time: {end_time - start_time:.2f} seconds")
    print(f"Final evaluation accuracy: {acc:.2f} %, Time for training: {elapsed_time:.2f} seconds")

