import pathlib
import project.utils as utils
from project.models.TIMM import TimmModel
from project.models.models import ResNet18_32x32
import torch
import torch.nn as nn
import tqdm
import yaml
import argparse
import optuna
import numpy as np
import random
import itertools
import copy
import functools
from collections import defaultdict
import project.augmentations as aug_lib
import torchvision

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}

def cifar10_resnet18_32x32(num_classes=10):
        model = ResNet18_32x32(num_classes=num_classes)
        model.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar10'])
        ])

        model.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar10'])
        ])
        model.name = "resnet18_32x32"
        return model

def build_model(model_name, real_data, pretrained):
    if model_name == "resnet18_32x32":
        model = cifar10_resnet18_32x32(real_data['num_classes'])
    else:
        model = TimmModel(model_name, real_data['num_classes'], pretrained=pretrained)
        model.encoder_grad(not pretrained)
        model.head_grad(True)
    return model


def train_baseline_conf(conf):
    dataset = conf['dataset']
    device = conf['device'] 
    exp = conf['exp']
    mode = conf['mode']
    root_path = pathlib.Path(conf['root_path'])
    data_path = root_path / conf['data_path']
    checkpoint_path = root_path / conf['checkpoint_path']
    generator_path = checkpoint_path / utils.generator_path(conf['generator_path'], dataset, exp)
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    num_epochs = conf['num_epochs']
    lr = conf['lr']
    weight_decay = conf['weight_decay']
    model_name = conf['model_name']
    pretrained = conf['pretrained']
    augmentations = conf['augmentations']
    shuffle = conf['shuffle']
    unfreeze_after = conf['unfreeze_after']
    experiment = conf['name']
    id = conf['id']
    save = conf['save']
    verbose = conf['verbose']

    aug = functools.partial(aug_lib.diff_augment, strategy=augmentations, param=aug_lib.ParamDiffAug())
    # Load the dataset
    real_data = utils.get_dataset(dataset, data_path)
    torch.backends.cudnn.benchmark = True

        
    if mode == "Real":
        loader = torch.utils.data.DataLoader(real_data['train'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=utils.imge_stack)
    elif mode == "Random":
        dim_z = 128
        config = {
            'dim_z': dim_z,  
            'resolution': 32,
            'G_attn': '0', 
            'n_classes': real_data['num_classes'],
            'G_shared': False, 
            }

        loader = utils.GeneratorDatasetLoader(torch.randn(len(real_data['train']), dim_z), real_data['label_train'], 'BigGAN', config, generator_path, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, device=device)
    else:
        anchors_path = checkpoint_path / utils.anchor_path(conf['anchors_path'], mode, dataset, exp)
        loader = utils.get_generator(anchors_path, generator_path, shuffle=shuffle, 
                                    num_workers=num_workers, batch_size=batch_size, device=device)
    eval_data = torch.utils.data.DataLoader(real_data['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack)


    print(f"Training {mode} {model_name}")
    model = build_model(model_name, real_data, pretrained).to(device)

    train_loader = utils.TransformLoader(loader, utils.get_transforms(model, mode, pretrained, device=device))
    eval_loader = utils.ImageLoader(eval_data, utils.get_transforms(model, 'Real', pretrained))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_epochs)
    # scaler = torch.cuda.amp.GradScaler()
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    best_model = None
    if verbose:
        epoch_iter = tqdm.trange(num_epochs, desc=mode)
    else:
        epoch_iter = range(num_epochs)
    test_acc = 0
    for epoch in epoch_iter:
        if epoch == int(num_epochs*unfreeze_after) and pretrained:
            model.encoder_grad(True)
        if verbose:
            epoch_iter.set_description(f"{mode} - training")
        train_acc, train_loss = utils.train(model, train_loader, optimizer, criterion, aug=aug)
        lr_scheduler.step()
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        if verbose:
            epoch_iter.set_description(f"{mode} - evaluating")
        test_acc_tmp, test_loss = utils.test(model, eval_loader, criterion)
        test_accs.append(test_acc_tmp)
        test_losses.append(test_loss)
        if test_acc_tmp > test_acc:
            best_model = model.state_dict()
        test_acc=test_acc_tmp
        if verbose:
            epoch_iter.set_description(f"{mode} - Done")
            epoch_iter.set_postfix(test_acc=f"{test_acc*100:.1f}%", train_acc=f"{train_acc*100:.1f}%", train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}")
    result = {
        "test_acc" : test_accs,
        "train_acc" : train_accs,
    }
    if save:
        model_path = checkpoint_path / f"{dataset}/{model.name}/pretrained_{pretrained}/{experiment}/{mode}_exp{exp}_{id}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "latest_state_dict": model.state_dict(),
            "best_state_dict": best_model,
            **conf,
            **result
        }, 
        model_path)
    return result


def get_baseline_model(conf, pre_trained=True):
    dataset = conf['dataset']
    device = conf['device']
    exp = conf['exp']
    mode = conf['mode']
    root_path = pathlib.Path(conf['root_path'])
    data_path = root_path / conf['data_path']
    checkpoint_path = root_path / conf['checkpoint_path']
    model_name = conf['model_name']
    drop_rate = conf['drop_rate']
    pretrained = conf['pretrained']
    experiment = conf['name']
    id = conf['id']
    real_data = utils.get_dataset(dataset, data_path)
    model = build_model(model_name, real_data, pretrained).to(device)
    if not pre_trained:
        return model

    model_path = checkpoint_path / f"{dataset}/{model.name}/pretrained_{pretrained}/{experiment}/{mode}_exp{exp}_{id}.pt"
    if model_path.exists():
        state_dict = torch.load(model_path)
        print(f"Loading {model_path}")
        model.load_state_dict(state_dict["latest_state_dict"])
        return model
    else:
        print(f"Training {model_path}")
        result = train_baseline_conf(conf)
        model.load_state_dict(result["best_state_dict"])
        return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        conf = yaml.safe_load(file)
    train_baseline_conf(conf)