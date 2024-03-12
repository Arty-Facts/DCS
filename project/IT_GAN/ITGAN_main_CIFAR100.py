import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from utils import get_dataset, get_network, DiffAugment, ParamDiffAug, epoch, get_time, save_image_tensor, distance_wb, match_loss, get_pretrained_networks, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import project.BigGAN as BigGAN
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model_train', type=str, default='ConvNet', help='model')
    parser.add_argument('--model_eval', type=str, default='ResNet18BN', help='model')
    parser.add_argument('--net_distribution', type=str, default='random', help='training net distribution, valid for ConvNet only')
    parser.add_argument('--acc_net_min', type=float, default=0, help='acc_net_min')
    parser.add_argument('--acc_net_max', type=float, default=1, help='acc_net_max')
    parser.add_argument('--split', type=str, default='0-20-0', help='x-y-z: exp x has total y classes, the current split class set z')
    parser.add_argument('--ratio_div', type=float, default=0, help='ratio_div')
    parser.add_argument('--Iteration', type=int, default=10000, help='Iterations to train z')
    parser.add_argument('--Epoch_evaltrain', type=int, default=200, help='epochs to train a network')
    parser.add_argument('--num_evalnet', type=int, default=3, help='train a number of networks per experiment')
    parser.add_argument('--lr_net', type=float, default=0.01, help='two stage: lr, lr/10') # start learning rate for training network with cos decrease schedule
    parser.add_argument('--lr_z', type=float, default=0.1, help='fixed learning rate for training z')
    parser.add_argument('--batch_train_z', type=int, default=500, help='batch size for sampling z, it should be a divisor of the image number per class per split')
    parser.add_argument('--batch_condense', type=int, default=500, help='batch size for sampling real images to condense knowledge')
    parser.add_argument('--batch_train_net', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--z_sample_mode', type=str, default='fixed', help='sample z batches in random or fixed correspondence')
    parser.add_argument('--match_mode', type=str, default='feat', help='match feature or gradient')
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
    G.eval()  

    for param in G.parameters():
        param.requires_grad = False

    mean_GAN = [0.5, 0.5, 0.5]
    std_GAN = [0.5, 0.5, 0.5]

    def renormalize(img):
        return torch.cat([(((img[:, 0] * std_GAN[0] + mean_GAN[0]) - mean[0]) / std[0]).unsqueeze(1),
                          (((img[:, 1] * std_GAN[1] + mean_GAN[1]) - mean[1]) / std[1]).unsqueeze(1),
                          (((img[:, 2] * std_GAN[2] + mean_GAN[2]) - mean[2]) / std[2]).unsqueeze(1)], dim=1)


    def generate(z, lab):
        num_max = 500 # Error occurs when batch size of G is large.
        num = z.shape[0]
        if num > num_max:
            img_syn = []
            for i in range(int(np.ceil(num / num_max))):
                img_syn.append(renormalize(G(z[i*num_max: (i+1)*num_max], lab[i*num_max: (i+1)*num_max])))
            return torch.cat(img_syn, dim=0)
        else:
            return renormalize(G(z, lab))




    ''' load GAN inversion z '''
    fpath = os.path.join(data_path, 'metasets', 'GANInversion_final_%s_ConvNet_lrz0.100_exp%d.pt' % (args.dataset, args.exp))
    print('use GAN inversion vectors: %s'%fpath)
    data_z = torch.load(fpath, map_location=device)
    z_inverse_all = deepcopy(data_z['z_inverse_all'])



    ''' construct the current data split '''
    num_classes_split = num_classes // args.total_splits
    np.random.seed(args.exp)
    classes_split = np.random.permutation(num_classes)[args.current_split*num_classes_split: (args.current_split+1)*num_classes_split]
    print('random seed: %d, the %d split of total %d split'%(args.exp, args.current_split, args.total_splits))
    print(classes_split)

    optm_split = []
    z_split = []
    idx_split = []
    img_split = [] # ep real images to z
    lab_split = []

    for c in classes_split: # c is the class id
        idx_c = indices_class[c]
        idx_split.append(deepcopy(idx_c))
        z_c = torch.tensor(deepcopy(z_inverse_all[idx_c].detach().clone()), dtype=torch.float, requires_grad=True, device=device)
        z_split.append(z_c)
        optm_c =  torch.optim.Adam([z_c], lr=args.lr_z, betas=[0.9, 0.999])
        optm_split.append(optm_c)
        img_split.append(images_all[idx_c])
        lab_split.append(labels_all[idx_c])


    # construct sub-test-set
    print('construct the sub-test-set for current class splits')
    images_test = []
    labels_test = []
    for i in range(len(dst_test)):
        img = torch.unsqueeze(dst_test[i][0], dim=0)
        lab = int(dst_test[i][1])
        if lab in classes_split:
            images_test.append(img)
            labels_test.append(lab)
    images_test = torch.cat(images_test, dim=0)
    labels_test = torch.tensor(labels_test, dtype=torch.long, device=device)
    dst_test_split = TensorDataset(images_test, labels_test)
    testloader = torch.utils.data.DataLoader(dst_test_split, batch_size=args.batch_train_net, shuffle=True, num_workers=0)
    print('images_test: ', images_test.shape)
    print('labels_test: ', labels_test.shape)

    criterion = nn.CrossEntropyLoss().to(device)


    ''' load network distribution '''
    if args.net_distribution == 'random': # use it
        networks = []
    else:
        print('loading pre-trained networks...')
        # networks = get_pretrained_networks(args.net_distribution, root_path, args.dataset, args.model_train, float(args.acc_net_min), float(args.acc_net_max))


    for itz in range(args.Iteration+1):

        ''' save and visualize '''
        if itz % 1000 == 0 or itz == args.Iteration:
            z_mean = np.mean([torch.mean(z_split[ic]).item() for ic in range(num_classes_split)])
            z_std = np.mean([torch.std(z_split[ic].reshape((-1))).item() for ic in range(num_classes_split)])
            z_grad = np.mean([torch.norm(z_split[ic].grad.detach()).item() for ic in range(num_classes_split)]) if itz>0 else 0
            print('z mean = %.4f, z std = %.4f, z.grad norm = %.6f' % (z_mean, z_std, z_grad))

            num_vis_pc = 10
            images_tosave = []
            for ic in range(num_classes_split):
                z_vis = z_split[ic][:num_vis_pc].detach()
                lab_vis = lab_split[ic][:num_vis_pc].detach()
                img_real_vis = img_split[ic][:num_vis_pc].detach()
                img_syn_vis = deepcopy(generate(z_vis, lab_vis).detach())
                images_tosave += [img_real_vis, img_syn_vis]

            save_name = os.path.join(save_path, 'EfficientGAN_final_vis_%s_%s_%s_%s_net%s_min%s_max%s_zbs%d_lrz%.3f_rdiv%.3f_split%s.png' % (args.dataset, args.model_train, args.match_mode, args.z_sample_mode, args.net_distribution, args.acc_net_min, args.acc_net_max, args.batch_train_z, args.lr_z, args.ratio_div, args.split))
            # save_name = os.path.join(save_path, 'EfficientGAN_vis_%s_%s_%s_%s_net%s_min%s_max%s_zbs%d_lrz%.3f_rdiv%.3f_split%s.png' % (args.dataset, args.model_train, args.match_mode, args.z_sample_mode, args.net_distribution, args.acc_net_min, args.acc_net_max, args.batch_train_z, args.lr_z, args.ratio_div, args.split))
            save_image_tensor(torch.cat(images_tosave, dim=0), mean, std, save_name, num_vis_pc)
            print('save to %s'%save_name)

            save_name = os.path.join(save_path, 'EfficientGAN_final_%s_%s_%s_%s_net%s_min%s_max%s_zbs%d_lrz%.3f_rdiv%.3f_split%s.pt'%(args.dataset, args.model_train, args.match_mode, args.z_sample_mode, args.net_distribution, args.acc_net_min, args.acc_net_max, args.batch_train_z, args.lr_z, args.ratio_div, args.split))
            # save_name = os.path.join(save_path, 'EfficientGAN_%s_%s_%s_%s_net%s_min%s_max%s_zbs%d_lrz%.3f_rdiv%.3f_split%s.pt'%(args.dataset, args.model_train, args.match_mode, args.z_sample_mode, args.net_distribution, args.acc_net_min, args.acc_net_max, args.batch_train_z, args.lr_z, args.ratio_div, args.split))
            torch.save({'z_split': z_split, 'idx_split': idx_split, 'classes_split': classes_split, 'train_iter': itz}, save_name)
            print('save to %s'%save_name)




        ''' evaluate GAN baseline '''

        if itz == 0:
            print('evaluate GAN baseline')
            lab_eval = [lab_split[ic].detach() for ic in range(num_classes_split)]
            lab_eval = deepcopy(torch.cat(lab_eval, dim=0))
            z_eval = torch.randn(size=(lab_eval.shape[0], dim_z), dtype=torch.float, requires_grad=False, device=device)


            def load_batch(idx):
                img = generate(z_eval[idx], lab_eval[idx])
                lab = lab_eval[idx]
                return img.detach(), lab.detach()


            # for args.model_eval in ['ResNet18BN']:
            for args.model_eval in ['ConvNet', 'ResNet18BN']:
                accs_test = []
                for eval_exp in range(args.num_evalnet):
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

                print('Evaluation: train z iter = %d evaluate %d %s test acc = %.4f std = %.4f\n'%(-1, len(accs_test), args.model_eval, np.mean(accs_test), np.std(accs_test)))
                print('Evaluation: train z iter = %d evaluate %d %s all results: '%(-1, len(accs_test), args.model_eval), accs_test)
                print('============================================================\n\n')






        ''' evaluate '''
        if itz % 5000 == 0 or itz == args.Iteration:
        # if (itz % 1000 == 0 or itz == args.Iteration) and itz > 0:

            z_eval = [z_split[ic].detach() for ic in range(num_classes_split)]
            z_eval = deepcopy(torch.cat(z_eval, dim=0))
            lab_eval = [lab_split[ic].detach() for ic in range(num_classes_split)]
            lab_eval = deepcopy(torch.cat(lab_eval, dim=0))


            def load_batch(idx):
                img = generate(z_eval[idx], lab_eval[idx])
                lab = lab_eval[idx]
                return img.detach(), lab.detach()


            # for args.model_eval in ['ResNet18BN']:
            for args.model_eval in ['ConvNet', 'ResNet18BN']:
                accs_test = []
                for eval_exp in range(args.num_evalnet):
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

        loss_divs_all = []
        loss_cond_all = []
        loss_all = []
        train_begin = time.time()

        for ic, c in enumerate(classes_split):
            # sample network
            if args.net_distribution == 'random':
                net = get_network(args.model_train, channel, num_classes, args.width_net, args.depth_net, args.act, args.normlayer, args.pooling, shape_img)
            else:
                net = deepcopy(networks[np.random.permutation(len(networks))[0]]).to(device)
            net.train() # To be verifed.
            net_parameters = list(net.parameters())

            if args.match_mode == 'feat':  # distribution matching
                for param in net.parameters():
                    param.requires_grad = False

            seed = int(time.time() * 1000) % 1000000
            idx = np.random.permutation(len(idx_split[ic]))[:args.batch_train_z]
            z_syn = z_split[ic][idx]
            lab_syn = lab_split[ic][idx].detach()
            img_syn = generate(z_syn, lab_syn)
            img_syn = DiffAugment(img_syn, args.diffaug_choice, seed=seed, param=param_diffaug)
            feat_syn = net.embed(img_syn)

            if args.ratio_div > 0.00001:
                img_real = img_split[ic][idx].detach()
                img_real = DiffAugment(img_real, args.diffaug_choice, seed=seed, param=param_diffaug)
                feat_real = net.embed(img_real).detach()
                loss_divs = torch.mean(torch.sum((feat_syn - feat_real) ** 2, dim=-1))
            else:
                loss_divs = torch.tensor(0.0).to(args.device)

            # sample a lage batch of real images img_c
            img_cond = get_images(c, args.batch_condense)
            img_cond = DiffAugment(img_cond, args.diffaug_choice, seed=seed, param=param_diffaug)
            lab_cond = torch.ones((args.batch_condense,), dtype=torch.long, requires_grad=False, device=device)*c

            # gradient matching
            if args.match_mode == 'grad': # gradient matching
                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                output_cond = net(img_cond)
                ls_cond = criterion(output_cond, lab_cond)
                gw_cond = torch.autograd.grad(ls_cond, net_parameters)
                gw_cond = list((_.detach().clone() for _ in gw_cond))

                loss_cond = match_loss(gw_syn, gw_cond, args)

            elif args.match_mode == 'feat': # distribution matching
                feat_cond = torch.mean(net.embed(img_cond).detach(), dim=0)  # averaged
                loss_cond = torch.sum((torch.mean(feat_syn, dim=0) - feat_cond) ** 2)

            else:
                loss_cond = 0
                exit('unknow args.match_mode')



            loss = args.ratio_div * loss_divs + (1 - args.ratio_div) * loss_cond
            # try to move out of the loop to speed up
            optm_split[ic].zero_grad()
            loss.backward()
            optm_split[ic].step()

            loss_divs_all.append(loss_divs.item())
            loss_cond_all.append(loss_cond.item())
            loss_all.append(loss.item())


        train_end = time.time()
        time_train = train_end - train_begin


        if itz % 10 == 0 or itz == args.Iteration:
            print('%s %s iter %05d time: %.1f loss: divs = %.4f * %.3f cond = %.4f * %.3f weighted_sum = %.4f'%(get_time(), args.split, itz, time_train, np.mean(loss_divs_all), args.ratio_div, np.mean(loss_cond_all), (1 - args.ratio_div), np.mean(loss_all)))


if __name__ == '__main__':
    main()

