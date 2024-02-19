import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from utils import get_dataset, get_network, DiffAugment, ParamDiffAug, epoch, get_time, save_image_tensor, distance_wb, match_loss

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model_train', type=str, default='ConvNet', help='model')
    parser.add_argument('--model_eval', type=str, default='ResNet18BN', help='model')
    parser.add_argument('--split', type=str, default='0-50-0', help='x-y-z: exp x has total y splits, the current split index z')
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
    if use_cuda:  # servers:
        root_path = '.'
    else:  # pc:
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
    #         try:
    #             torch.save({'net': net, 'acc_train': acc_train, 'acc_test': acc_test}, PreTrnNetPath, _use_new_zipfile_serialization=False)
    #         except Exception:
    #             torch.save({'net': net, 'acc_train': acc_train, 'acc_test': acc_test}, PreTrnNetPath)
    #         print('save to %s'%PreTrnNetPath)
    #
    #
    # print('Pre-train ConvNet finished')




    ''' Concatenate GAN inversion z vectors '''
    # dim_z = 128
    # z_inverse_all = torch.zeros(size=(num_train, dim_z), dtype=torch.float, requires_grad=False, device=device)
    # save_name = os.path.join(save_path, 'GANInversion_final_%s_%s_lrz0.100_exp%d.pt'%(args.dataset, args.model_train, args.exp))
    # for i in range(args.total_splits):
    #     fpath = os.path.join(save_path, 'GANInversion_final_%s_%s_lrz0.100_split%d-%d-%d.pt'%(args.dataset, args.model_train, args.exp, args.total_splits, i))
    #     print('processing %s'%fpath)
    #     datax = torch.load(fpath, map_location=device)
    #     z_inv = datax['z_split']
    #     idxs = datax['idx_split']
    #     # for c in range(num_classes):
    #     #     z_inverse_all[idx_split[c]] = z_split[c]
    #
    #     train_iter = datax['train_iter']
    #     if train_iter != 5000:
    #         exit('z is not trained for 5000 epochs in %s' % fpath)
    #
    #     for (z, idx) in zip(z_inv, idxs):
    #         z_inverse_all[idx] = z
    #
    #
    # try:
    #     torch.save({'z_inverse_all': z_inverse_all}, save_name, _use_new_zipfile_serialization=False)
    # except Exception:
    #     torch.save({'z_inverse_all': z_inverse_all}, save_name)
    #
    # print('save to ', save_name)
    #



    ''' Concatenate EfficientGAN z vectors '''
    dim_z = 128
    total_splits = 4
    z_eff_all = torch.zeros(size=(num_train, dim_z), dtype=torch.float, requires_grad=False, device=device)
    save_name = os.path.join(save_path, 'EfficientGAN_final_%s_%s_lrz%.3f_exp%d.pt'%(args.dataset, args.model_train, 0.001, args.exp))

    for current_split in range(total_splits):
        split = '%d-%d-%d' % (args.exp, total_splits, current_split)
        # fpath = os.path.join(os.path.join(root_path, 'results', 'EfficientGAN'), 'EfficientGAN_final_%s_ConvNet_feat_fixed_netmin_max_min0.7_max1.0_zbs1250_lrz0.001_rdiv0.000_split%s.pt' % (args.dataset, split))
        fpath = os.path.join(os.path.join(root_path, 'results', 'EfficientGAN'), 'EfficientGAN_final_%s_ConvNet_feat_fixed_netmin_max_min0.4_max1.0_zbs500_lrz0.001_rdiv0.000_split%s.pt' % (args.dataset, split))
        print('load EfficientGAN: %s' % fpath)

        datax = torch.load(fpath, map_location=device)

        train_iter = datax['train_iter']
        if train_iter != 5000:
            exit('z is not trained for 5000 epochs in %s' % fpath)

        z_eff = datax['z_split']
        idxs = datax['idx_split']
        for (z, idx) in zip(z_eff, idxs):
            z_eff_all[idx] = z

    try:
        torch.save({'z_eff_all': z_eff_all}, save_name, _use_new_zipfile_serialization=False)
    except Exception:
        torch.save({'z_eff_all': z_eff_all}, save_name)

    print('save to ', save_name)









if __name__ == '__main__':
    main()







