import torch
from torch import nn 
import numpy as np
from torchvision.utils import save_image, make_grid
import os
import time

from generative_distillation.gan_model import Generator, Discriminator
from generative_distillation.augment import diffaug
from common import define_model
from generative_distillation.gen_utils import AverageMeter, accuracy, calc_gradient_penalty, test, rand_bbox, matchloss, train_match_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand, mode='gan'):
    '''The main training function for the generator
    '''
    generator.train()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if mode == 'match': 
        model = define_model(args, args.num_classes).cuda()
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.cuda()
        lab_real = lab_real.cuda()

        # train the generator
        discriminator.eval()
        optim_g.zero_grad()

        # obtain the noise with one-hot class labels
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_real] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()

        img_syn = generator(noise)
        gen_source, gen_class = discriminator(img_syn)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, lab_real)
        gen_loss = - gen_source + gen_class

        if mode == 'match':
            train_match_model(args, model, optim_model, trainloader, criterion, aug_rand)
            if args.match_aug:
                img_aug = aug(torch.cat([img_real, img_syn]))
                match_loss = matchloss(args, img_aug[:args.batch_size], img_aug[args.batch_size:], lab_real, lab_real, model)# * args.match_coeff
            else:
                match_loss = matchloss(args, img_real, img_syn, lab_real, lab_real, model)# * args.match_coeff

            gen_loss = gen_loss + match_loss
        
        gen_loss.backward()
        optim_g.step()

        # train the discriminator
        discriminator.train()
        optim_d.zero_grad()
        lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()
        lab_syn = lab_syn.cuda()

        with torch.no_grad():
            img_syn = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(img_syn)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, lab_syn)

        disc_real_source, disc_real_class = discriminator(img_real)
        acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, lab_real)

        gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn)

        disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
        disc_loss.backward()
        optim_d.step()

        gen_losses.update(gen_loss.item())
        disc_losses.update(disc_loss.item())
        top1.update(acc1.item())
        top5.update(acc5.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg, disc_losses.val, disc_losses.avg, top1.val, top1.avg)
            )

def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model = define_model(args, args.num_classes, e_model=e_model).cuda()
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0
        total_gen_time = 0.0
        total_batches_count = 0
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size):
                # obtain pseudo samples with the generator
                torch.cuda.synchronize()
                start_time = time.time()
                if args.batch_size == args.num_classes:
                    lab_syn = torch.randperm(args.num_classes)
                else:
                    lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
                lab_onehot = torch.zeros((args.batch_size, args.num_classes))
                lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
                noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                    img_syn = generator(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)
                torch.cuda.synchronize()
                end_time = time.time()
                total_gen_time += end_time - start_time
                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()
                total_batches_count += 1 
            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        print(f'Average generation time per batch: {total_gen_time / total_batches_count:.4f} seconds')

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5

def start_dim(args, trainset, testset):

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    generator = Generator(args).to(DEVICE)
    discriminator = Discriminator(args).to(DEVICE)

    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)
    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))

    for epoch in range(args.epochs):

        if args.tag == 'test':
            generator = Generator(args).cuda()
            generator.load_state_dict(torch.load(args.pretrain_weight)['generator'])
            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)

            for e_idx, e_model in enumerate(args.eval_model):
                print('Evaluation for {}: Top1: {:.3f}, Top5: {:.3f}'.format(e_model, top1s[e_idx], top5s[e_idx]))
            break

        generator.train()
        discriminator.train()

        if args.tag == 'gan': 
            train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand)
        elif args.tag == 'match': 
            train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand, mode='match')
            pass
       

        generator.eval()
        test_label = torch.tensor(list(range(10)) * 10)
        test_noise = torch.normal(0, 1, (100, 100))
        lab_onehot = torch.zeros((100, args.num_classes))
        lab_onehot[torch.arange(100), test_label] = 1
        test_noise[torch.arange(100), :args.num_classes] = lab_onehot[torch.arange(100)]
        test_noise = test_noise.cuda()
        test_img_syn = (generator(test_noise) + 1.0) / 2.0
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(test_img_syn, os.path.join(args.output_dir, 'outputs/img_{}.png'.format(epoch)))
        generator.train()

        if (epoch + 1) % args.eval_interval == 0:
            model_dict = {'generator': generator.state_dict(),
                          'discriminator': discriminator.state_dict(),
                          'optim_g': optim_g.state_dict(),
                          'optim_d': optim_d.state_dict()}
            torch.save(
                model_dict,
                os.path.join(args.output_dir, 'model_dict_{}.pth'.format(epoch)))
            print("img and data saved!")

            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
            for e_idx, e_model in enumerate(args.eval_model):
                if top1s[e_idx] > best_top1s[e_idx]:
                    best_top1s[e_idx] = top1s[e_idx]
                    best_top5s[e_idx] = top5s[e_idx]
                    best_epochs[e_idx] = epoch
                print('Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(e_model, best_epochs[e_idx], best_top1s[e_idx], best_top5s[e_idx]))