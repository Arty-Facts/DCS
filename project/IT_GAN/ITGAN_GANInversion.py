import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from utils import get_dataset, get_network, DiffAugment, ParamDiffAug, epoch, get_time, save_image_tensor, distance_wb, match_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import project.BigGAN as BigGAN
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model_train', type=str, default='ConvNet', help='model')
    parser.add_argument('--model_eval', type=str, default='ResNet18BN', help='model')
    parser.add_argument('--split', type=str, default='0-1-0', help='x-y-z: exp x has total y splits, the current split index z')
    parser.add_argument('--Iteration', type=int, default=10000, help='Iterations to train z')
    parser.add_argument('--Epoch_evaltrain', type=int, default=200, help='epochs to train a network')
    parser.add_argument('--lr_net', type=float, default=0.01, help='two stage: lr, lr/10') # start learning rate for training network with cos decrease schedule
    parser.add_argument('--lr_z', type=float, default=0.1, help='fixed learning rate for training z')
    parser.add_argument('--batch_train_z', type=int, default=50, help='batch size for sampling z, it should be a divisor of the image number per class per split')
    parser.add_argument('--batch_train_net', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--exp_mode', type=str, default='train', help='train or evaluate z')
    parser.add_argument('--diffaug_choice', type=str, default='Auto', help='diffaug_choice')

    # for ConvNet only
    parser.add_argument('--width_net', type=int, default=128, help='width')
    parser.add_argument('--depth_net', type=int, default=3, help='depth')
    parser.add_argument('--act', type=str, default='relu', help='act')
    parser.add_argument('--normlayer', type=str, default='instancenorm', help='normlayer')
    parser.add_argument('--pooling', type=str, default='avgpooling', help='pooling')

    args = parser.parse_args()

    # for augmentation
    param_diffaug = ParamDiffAug()

    if args.diffaug_choice == 'Auto':
        if args.dataset in ['MNIST', 'SVHN']:
            args.diffaug_choice = 'color_crop_cutout_scale_rotate'
        elif args.dataset in ['FashionMNIST', 'CIFAR10', 'CIFAR100']:
            args.diffaug_choice = 'color_crop_cutout_flip_scale_rotate'
        else:
            exit('Auto diffaug_choice is not defined for dataset: %s' % args.dataset)
    else:
        args.diffaug_choice = 'None'

    # gpu usage
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    if use_cuda:  
        root_path = '.'
    else: # other folder
        root_path = '.' 
    data_path = os.path.join(root_path, '../data')
    print('gpu number = %d' % (torch.cuda.device_count()))

    args.dis_metric = 'ours' # gradient matching metric, 'ours' is from DC.
    args.device = device # gradient matching metric

    # experiment index
    tokens = str(args.split).split('-')
    args.exp = int(tokens[0])
    args.total_splits = int(tokens[1])
    args.current_split = int(tokens[2])


    print('args:')
    print(args.__dict__)
    print('param_diffaug:')
    print(param_diffaug.__dict__)
    print('device: ', device)

    channel, shape_img, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    num_train = dst_train.__len__()
    print('dst_train length: ', num_train)


    save_path = os.path.join(root_path, 'results', 'EfficientGAN')
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    ''' load data '''
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0)
    images_all = images_all.to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    print('Total dataset images_all shape: ', images_all.shape)
    print('Total dataset images_all mean = [%.4f, %.4f, %.4f], std = [%.4f, %.4f, %.4f]' % (torch.mean(images_all[:, 0]), torch.mean(images_all[:, 1]), torch.mean(images_all[:, 2]),
                                                                              torch.std(images_all[:, 0]), torch.std(images_all[:, 1]), torch.std(images_all[:, 2])))

    def get_images(c, num):  # get random num images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:num]
        return images_all[idx_shuffle]



    ''' Pre-train ConvNet '''
    # print('Pre-train ConvNet')
    # PreTrnNetPath = os.path.join(save_path, 'ConvNet_Pretrained_%s_exp%d.pt' % (args.dataset, args.exp))
    # num_evaltrain = int(images_all.shape[0])
    # net = get_network('ConvNet', channel, num_classes, args.width_net, args.depth_net, args.act, args.normlayer, args.pooling, shape_img)
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)  # no cuda version
    # # scheduler = CosineAnnealingLR(optimizer_net, args.Epoch_evaltrain, 0.0001)
    #
    # def load_batch(idx):
    #     return images_all[idx].detach(), labels_all[idx].detach()
    #
    # for ep_eval in range(args.Epoch_evaltrain + 1):
    #     train_begin = time.time()
    #     net.train()
    #     idx_rand = np.random.permutation(num_evaltrain)
    #     acc_train = []
    #     loss_train = []
    #
    #     if ep_eval == args.Epoch_evaltrain // 2:
    #         optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net / 10, momentum=0.9, weight_decay=0.0005)  # no cuda version
    #
    #     for it in range(int(np.ceil(num_evaltrain // args.batch_train_net))):
    #         img, lab = load_batch(idx_rand[it * args.batch_train_net: (it + 1) * args.batch_train_net])
    #         img = DiffAugment(img, args.diffaug_choice, param=param_diffaug)
    #         output = net(img.float())
    #         loss = criterion(output, lab)
    #
    #         optimizer_net.zero_grad()
    #         loss.backward()
    #         optimizer_net.step()
    #
    #         acc_train.append(np.mean(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())))
    #         loss_train.append(loss.item())
    #
    #     train_end = time.time()
    #     time_train = train_end - train_begin
    #     # scheduler.step()
    #
    #     if ep_eval % 5 == 0 or ep_eval == args.Epoch_evaltrain:
    #         loss_test, acc_test, acc_separate = epoch('test', 0, testloader, net, optimizer_net, criterion, device=device, flag_print=False)
    #         print('%s %s epoch %d/%d  time = %.1f  lr = %.4f  train_loss = %.4f  train_acc = %.4f  test_acc = %.4f' % (
    #         get_time(), args.split, ep_eval, args.Epoch_evaltrain, time_train, optimizer_net.param_groups[0]['lr'], np.mean(loss_train), np.mean(acc_train), acc_test))
    #
    #         torch.save({'net': net, 'acc_train': acc_train, 'acc_test': acc_test}, PreTrnNetPath, _use_new_zipfile_serialization=False)
    #
    # print('Pre-train ConvNet finished')



    ''' load pre-trained feature extractor '''
    PreTrnNetPath = os.path.join(data_path, 'metasets', '%s_Pretrained_%s_exp%d.pt' % (args.model_train, args.dataset, args.exp))
    try:
        net_pretrn = torch.load(PreTrnNetPath, map_location=device)['net']
    except Exception:
        net_pretrn = torch.jit.load(PreTrnNetPath, map_location=device)['net']



    ''' load pre-trained GAN '''
    weight_path = os.path.join(data_path, 'metasets', 'G_Pretrained_%s_exp%d.pth' % (args.dataset, args.exp))
    print('use G model: ', weight_path)

    dim_z = 128
    G = BigGAN.Generator(G_ch=64, dim_z=dim_z, bottom_width=4, resolution=32,
                         G_kernel_size=3, G_attn='0', n_classes=num_classes,
                         num_G_SVs=1, num_G_SV_itrs=1,
                         G_shared=False, shared_dim=0, hier=False,
                         cross_replica=False, mybn=False,
                         G_activation=nn.ReLU(inplace=False),
                         G_lr=2e-4, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                         BN_eps=1e-5, SN_eps=1e-08, G_mixed_precision=False, G_fp16=False,
                         G_init='N02', skip_init=False, no_optim=False,
                         G_param='SN', norm_style='bn').to(device)
    G.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    G.eval()  # Train? My conclusion is that .eval() is good for pretrained weights and mean/std, but .train() is good for random weights.
    # G.train()
    for param in G.parameters():
        param.requires_grad = False

    mean_GAN = [0.5, 0.5, 0.5]
    std_GAN = [0.5, 0.5, 0.5]

    def renormalize(img):
        return torch.cat([(((img[:, 0] * std_GAN[0] + mean_GAN[0]) - mean[0]) / std[0]).unsqueeze(1),
                          (((img[:, 1] * std_GAN[1] + mean_GAN[1]) - mean[1]) / std[1]).unsqueeze(1),
                          (((img[:, 2] * std_GAN[2] + mean_GAN[2]) - mean[2]) / std[2]).unsqueeze(1)], dim=1)





    ''' construct the current data split '''
    # shuffle the indices_class for each experiment
    np.random.seed(args.exp)
    for c in range(num_classes):
        np.random.shuffle(indices_class[c])

    print('random seed: ', args.exp)
    # for c in range(num_classes):
    #     print('shuffled indices_class %d: %s'%(c, indices_class[c][]))


    optm_split = []
    z_split = []
    idx_split = []
    img_split = [] # ep real images to z
    lab_split = []
    num_imgpercls = num_train//num_classes//args.total_splits

    for c in range(num_classes):
        idx_c = indices_class[c][args.current_split*num_imgpercls:(args.current_split+1)*num_imgpercls]
        idx_split.append(deepcopy(idx_c))
        z_c = torch.randn(size=(len(idx_c), dim_z), dtype=torch.float, requires_grad=True, device=device)
        z_split.append(z_c)
        # optm_c =  torch.optim.SGD([z_c, ], lr=args.lr_z, momentum=0.5)
        optm_c =  torch.optim.Adam([z_c], lr=args.lr_z, betas=[0.9, 0.999])
        optm_split.append(optm_c)
        img_split.append(images_all[idx_c])
        lab_split.append(labels_all[idx_c])




    ''' train z '''

    for itz in range(args.Iteration+1):

        ''' save and visualize '''
        if itz % 1000 == 0 or itz == args.Iteration:
            z_mean = np.mean([torch.mean(z_split[c]).item() for c in range(num_classes)])
            z_std = np.mean([torch.std(z_split[c].reshape((-1))).item() for c in range(num_classes)])
            z_grad = np.mean([torch.norm(z_split[c].grad.detach()).item() for c in range(num_classes)]) if itz>0 else 0
            print('z mean = %.4f, z std = %.4f, z.grad norm = %.6f' % (z_mean, z_std, z_grad))

            num_vis_pc = 10
            images_tosave = []
            for c in range(num_classes):
                z_vis = z_split[c][:num_vis_pc].detach()
                lab_vis = lab_split[c][:num_vis_pc].detach()
                img_real_vis = img_split[c][:num_vis_pc].detach()
                img_syn_vis = deepcopy(renormalize(G(z_vis, lab_vis)).detach())
                images_tosave += [img_real_vis, img_syn_vis]

            save_name = os.path.join(save_path, 'GANInversion_final_vis_%s_%s_lrz%.3f_split%s.png'%(args.dataset, args.model_train, args.lr_z, args.split))
            save_image_tensor(torch.cat(images_tosave, dim=0), mean, std, save_name, num_vis_pc)
            print('save to %s'%save_name)

            save_name = os.path.join(save_path, 'GANInversion_final_%s_%s_lrz%.3f_split%s.pt'%(args.dataset, args.model_train, args.lr_z, args.split))
            torch.save({'z_split': z_split, 'idx_split': idx_split, 'train_iter': itz}, save_name)
            print('save to %s'%save_name)




        ''' evaluate '''
        if itz % 5000 == 0 or itz == args.Iteration:
        # if (itz % 1000 == 0 or itz == args.Iteration) and itz > 0:

            z_eval = [z_split[c].detach() for c in range(num_classes)]
            z_eval = deepcopy(torch.cat(z_eval, dim=0))
            lab_eval = [lab_split[c].detach() for c in range(num_classes)]
            lab_eval = deepcopy(torch.cat(lab_eval, dim=0))


            def load_batch(idx):
                img = renormalize(G(z_eval[idx], lab_eval[idx]))
                lab = lab_eval[idx]
                return img.detach(), lab.detach()


            for args.model_eval in ['ConvNet', 'ResNet18BN']:
            # for args.model_eval in ['ResNet18BN']:
                accs_test = []
                for eval_exp in range(3):
                    print('--------------------------------------------------------------')
                    print('train z iter: %d eval_exp: %d'%(itz, eval_exp))
                    print('evaluate z train %s' % (args.model_eval))
                    print('z_eval: ', z_eval.shape)
                    print('lab_eval: ', lab_eval.shape)
                    print('args.batch_train_net: ', args.batch_train_net)
                    print('args.Epoch_evaltrain: ', args.Epoch_evaltrain)
                    print('args.lr_net: ', args.lr_net)
                    num_evaltrain = int(z_eval.shape[0])
                    print('num_evaltrain: ', num_evaltrain)

                    net = get_network(args.model_eval, channel, num_classes, args.width_net, args.depth_net, args.act, args.normlayer, args.pooling, shape_img)
                    criterion = nn.CrossEntropyLoss().to(device)
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)  # no cuda version
                    # scheduler = CosineAnnealingLR(optimizer_net, args.Epoch_evaltrain, 0.0001)


                    for ep_eval in range(args.Epoch_evaltrain+1):
                        train_begin = time.time()
                        net.train()
                        idx_rand = np.random.permutation(num_evaltrain)
                        acc_train = []
                        loss_train = []

                        if ep_eval == args.Epoch_evaltrain//2:
                            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net/10, momentum=0.9, weight_decay=0.0005)  # no cuda version

                        for it in range(int(np.ceil(num_evaltrain // args.batch_train_net))):
                            img, lab = load_batch(idx_rand[it*args.batch_train_net: (it+1)*args.batch_train_net])
                            img = DiffAugment(img, args.diffaug_choice, param=param_diffaug)
                            output = net(img.float())
                            loss = criterion(output, lab)

                            optimizer_net.zero_grad()
                            loss.backward()
                            optimizer_net.step()

                            acc_train.append(np.mean(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())))
                            loss_train.append(loss.item())

                        train_end = time.time()
                        time_train = train_end - train_begin
                        # scheduler.step()

                        if ep_eval%20 == 0 or ep_eval == args.Epoch_evaltrain:
                            loss_test, acc_test, acc_separate = epoch('test', 0, testloader, net, optimizer_net, criterion, device=device, flag_print=False)
                            print('%s %s epoch %d/%d  time = %.1f  lr = %.4f  train_loss = %.4f  train_acc = %.4f  test_acc = %.4f' % (get_time(), args.split, ep_eval, args.Epoch_evaltrain, time_train, optimizer_net.param_groups[0]['lr'], np.mean(loss_train), np.mean(acc_train), acc_test))

                    accs_test.append(acc_test)

                print('Evaluation: train z iter = %d evaluate %d %s test acc = %.4f std = %.4f\n'%(itz, len(accs_test), args.model_eval, np.mean(accs_test), np.std(accs_test)))
                print('Evaluation: train z iter = %d evaluate %d %s all results: '%(itz, len(accs_test), args.model_eval), accs_test)
                print('============================================================\n\n')



        ''' train '''

        loss_feat_all = []
        loss_pixel_all = []
        loss_all = []
        train_begin = time.time()

        for c in range(num_classes):
            # sample network
            net_pretrn.train() # To be verifed.

            for bc in range(int(np.ceil(num_imgpercls//args.batch_train_z))): # args.batch_train_z should be a divisor of num_imgpercls
                idx = np.arange(bc * args.batch_train_z, (bc + 1) * args.batch_train_z)
                z_syn = z_split[c][idx]
                lab_syn = lab_split[c][idx].detach()
                img_syn = renormalize(G(z_syn, lab_syn))
                feat_syn = net_pretrn.embed(img_syn)

                img_real = img_split[c][idx].detach()
                feat_real = net_pretrn.embed(img_real).detach()

                loss_feat = torch.mean((feat_syn - feat_real) ** 2)
                loss_pixel = torch.mean((img_syn - img_real) ** 2)
                loss = loss_feat + loss_pixel

                optm_split[c].zero_grad()
                loss.backward()
                optm_split[c].step()

                loss_feat_all.append(loss_feat.item())
                loss_pixel_all.append(loss_pixel.item())
                loss_all.append(loss.item())


        train_end = time.time()
        time_train = train_end - train_begin


        if itz % 10 == 0 or itz == args.Iteration:
            print('%s %s iter %05d time: %.1f loss: feat = %.4f pixel = %.4f sum = %.4f'%(get_time(), args.split, itz, time_train, np.mean(loss_feat_all), np.mean(loss_pixel_all), np.mean(loss_all)))








if __name__ == '__main__':
    main()

