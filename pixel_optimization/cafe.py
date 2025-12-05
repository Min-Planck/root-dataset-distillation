import torch
import torch.nn as nn
import numpy as np
import time
import copy 

from common import define_model
from pixel_optimization.pix_utils import get_images, get_loops, update_model, evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def criterion_middle(real_feature, syn_feature):
    MSE_Loss = nn.MSELoss(reduction='sum')
    shape_real = real_feature.shape
    real_feature = torch.mean(real_feature.view(10, shape_real[0] // 10, *shape_real[1:]), dim=1)

    shape_syn = syn_feature.shape
    syn_feature = torch.mean(syn_feature.view(10, shape_syn[0] // 10, *shape_syn[1:]), dim=1)

    return MSE_Loss(real_feature, syn_feature)

def condensation(
        args,
        channel,
        img_size,
        inner_loop, 
        eval_model,
        loss_fn, 
        criterion_sum,
        optimizer_img, 
        data_syn,
        target_syn,
        indices_class, 
        images_all): 
    
    loss_avg = 0.0
    outer_acc_watcher = []
    innter_acc_wathcher = []

    inner_loop_cnt = 0
    outer_loop_cnt = 0

    while True: 
        loss = torch.tensor(0.0).to(DEVICE)
        optimizer_net = torch.optim.SGD(eval_model.parameters(), lr=args.lr)
        optimizer_net.zero_grad()

        loss_avg = 0
        loss_kai = 0
        loss_middle_item = 0


        img_real_gather = []
        img_syn_gather = []
        lab_real_gather = []
        lab_syn_gather = []

        for c in range(args.num_classes):
            img_real = get_images(indices_class, images_all, c, args.batch_size)
            img_syn = data_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, img_size[0], img_size[1]))

            target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=DEVICE) * c
            target_syn = torch.ones((args.ipc,), dtype=torch.long, device=DEVICE) * c

            img_real_gather.append(img_real)
            lab_real_gather.append(target_real)
            img_syn_gather.append(img_syn)
            lab_syn_gather.append(target_syn)

        img_real_gather = torch.stack(img_real_gather, dim=0).reshape(args.batch_size * 10, channel, img_size[0], img_size[1])
        img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(args.ipc * 10, channel, img_size[0], img_size[1])
        lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(args.batch_size * 10)
        lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(args.ipc * 10)

        output_real, feature_r = eval_model(img_real_gather, return_all_layers=True)
        output_syn, feature_s = eval_model(img_syn_gather, return_all_layers=True)
        
        feature_real = feature_r[::-1]
        feature_syn = feature_s[::-1]
        loss_middle = args.fourth_weight * criterion_middle(feature_real[-1], feature_syn[-1]) + args.third_weight * criterion_middle(feature_real[-2], feature_syn[-2]) + args.second_weight * criterion_middle(feature_real[-3], feature_syn[-3]) + args.first_weight * criterion_middle(feature_real[-4], feature_syn[-4])
        loss_real = loss_fn(output_real, lab_real_gather)

        loss += loss_middle + loss_real
        fr = feature_real[0]                             
        fr = torch.flatten(fr, start_dim=1)              # [1280, C*H*W]
        fr = fr.view(args.num_classes, -1, fr.shape[1])  # [10, 128, C*H*W]
        last_real_feature = fr.mean(dim=1)

        fs = feature_syn[0]                            
        fs = torch.flatten(fs, start_dim=1)             
        fs = fs.view(args.num_classes, -1, fs.shape[1])  
        last_syn_feature = fs.mean(dim=1)

        output = torch.mm(last_real_feature, last_syn_feature.t())


        loss_output = criterion_middle(last_syn_feature, last_real_feature) + args.inner_weight * criterion_sum(output, lab_real_gather)
        loss += loss_output

        loss.backward()
        optimizer_img.step()
        optimizer_img.zero_grad()

        loss_avg += loss.item()
        loss_kai += loss_output.item()
        loss_middle_item += loss_middle.item()


        outer_acc = 0
        for c in range(args.num_classes):
            img_real = get_images(indices_class, images_all, c, args.batch_size)
            target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=DEVICE) * c

            output = eval_model(img_real)
            outer_acc += (output.argmax(dim=1) == target_real).sum().item()

        outer_acc /= args.num_classes
        outer_acc_watcher.append(outer_acc)
        outer_loop_cnt += 1

        if len(outer_acc_watcher) == 10:
            if max(outer_acc_watcher) - min(outer_acc_watcher) < args.lambda_1:
                outer_acc_watcher = list()
                outer_loop_cnt = 0
                outer_acc = 0.0
                break

            else:
                outer_acc_watcher.pop(0)

        image_syn_train, label_syn_train = copy.deepcopy(data_syn.detach()), copy.deepcopy(target_syn.detach())
        inner_acc_watcher = list()
        acc_syn_innter_watcher = list()
        
        inner_cnt = 0
        acc_test = 0
        while True:
            acc_syn = update_model(optimizer_net, inner_loop, loss_fn, eval_model, image_syn_train, label_syn_train)
            acc_syn_innter_watcher.append(acc_syn)

            for c in range(args.num_classes):
                img_real = get_images(indices_class, images_all, c, args.batch_size)
                target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=DEVICE) * c

                output = eval_model(img_real)
                acc_test += (target_real == output.argmax(dim=1)).sum().item() / args.batch_size

            acc_test /= args.num_classes
            inner_acc_watcher.append(acc_test)

            inner_cnt += 1
            if len(inner_acc_watcher) == 10:
                if max(inner_acc_watcher) - min(inner_acc_watcher) < args.lambda_2:
                    inner_acc_watcher = list()
                    inner_cnt = 0
                    acc_test = 0
                    break
                else:
                    inner_acc_watcher.pop(0)
        loss_avg /= (args.num_classes * args.ipc)
        
        if outer_loop_cnt % args.print_freq == 0: 
            print(f"Step {outer_loop_cnt + 1}, Loss: {loss_avg:.4f}")
            model_save_name = f'{args.eval_model[0]}_ipc{args.ipc}_step{outer_loop_cnt}.pt'
            path = f'{args.output_dir}/outputs/{model_save_name}'

            torch.save(data_syn, path)
        if outer_loop_cnt == args.epochs - 1:
            break

def start_cafe(args, trainset, testset): 
    ipc = args.ipc 
    _, inner_loop = get_loops(ipc)

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
    criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)


    start_time = time.time()

    eval_model = define_model(args, args.num_classes, 'single').to(DEVICE)

    condensation(
        args,
        channel,
        img_size,
        inner_loop, 
        eval_model,
        loss_fn, 
        criterion_sum,
        optimizer_img, 
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
    print(f'Algorithm: DC')
    print(f"Total condensation time: {end_time - start_time:.2f} seconds")
    print(f"Final evaluation accuracy: {acc:.2f} %, Time for training: {elapsed_time:.2f} seconds")

