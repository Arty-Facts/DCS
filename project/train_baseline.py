import pathlib
import project.utils as utils
from project.models.TIMM import TimmModel
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

def train_strategy_conf(conf):
    # torch.autograd.set_detect_anomaly(True)
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
    drop_rate = conf['drop_rate']
    pretrained = conf['pretrained']
    augmentations = conf['augmentations']
    shuffle = conf['shuffle']
    unfreeze_after = conf['unfreeze_after']
    experiment = conf['name']
    id = conf['id']
    strategy_name = conf['strategy']['name']
    save = conf['save']
    verbose = conf['verbose']

    aug = functools.partial(aug_lib.diff_augment, strategy=augmentations, param=aug_lib.ParamDiffAug())
    # Load the dataset
    real_data = utils.get_dataset(dataset, data_path)
    torch.backends.cudnn.benchmark = True

        
    if mode == "Random":
        dim_z = 128
        config = {
            'dim_z': dim_z,  
            'resolution': 32,
            'G_attn': '0', 
            'n_classes': real_data['num_classes'],
            'G_shared': False, 
            }

        generator = utils.GeneratorDatasetLoader(torch.randn(len(real_data['train']), dim_z), real_data['label_train'], 'BigGAN', config, generator_path, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, device=device, use_cache=strategy_name=='Static')
    else:
        anchors_path = checkpoint_path / utils.anchor_path(conf['anchors_path'], mode, dataset, exp)
        generator = utils.get_generator(anchors_path, generator_path, shuffle=shuffle, 
                                    num_workers=num_workers, batch_size=batch_size, device=device, use_cache=strategy_name=='Static')
    
    eval_data = utils.DeviceDataLoader(real_data['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)

    train_data = utils.DeviceDataLoader(real_data['train'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, device=device)

    print(f"Training {mode} {model_name}")
    model = TimmModel(model_name, real_data['num_classes'], drop_rate=drop_rate, pretrained=pretrained).to(device)
    model.encoder_grad(not pretrained)
    model.head_grad(True)
    transform_gen = utils.get_transforms(model, mode, pretrained)
    transform_real = utils.get_transforms(model, 'Real', pretrained)
    gen_loader = utils.TransformLoader(generator, transform_gen)
    eval_loader = utils.TransformLoader(eval_data, transform_real)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(loader)*num_epochs)
    # scaler = torch.cuda.amp.GradScaler()
    strategy = utils.get_strategy(conf['strategy'], 
                                  generator=generator,
                                  model=model, 
                                  real_data=train_data,
                                  )


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

        strategy(verbose=False)

        if epoch == int(num_epochs*unfreeze_after) and pretrained:
            model.encoder_grad(True)
        if verbose:
            epoch_iter.set_description(f"{mode} - training")
        train_acc, train_loss = utils.train(model, gen_loader, optimizer, criterion, aug=aug)
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
        model_path = checkpoint_path / f"{dataset}/{model.name}/pretrained_{pretrained}/{experiment}/{mode}_{strategy_name}_exp{exp}_acc{int(max(test_accs)*1000)}_id{id}.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "latest_state_dict": model.state_dict(),
            "best_state_dict": best_model,
            "anchors": generator.anchors.detach().clone().cpu(),
            "labels": generator.labels.detach().clone().cpu(),
            **conf,
            **result
        }, 
        model_path)
    return result

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
    drop_rate = conf['drop_rate']
    pretrained = conf['pretrained']
    augmentations = conf['augmentations']
    shuffle = conf['shuffle']
    unfreeze_after = conf['unfreeze_after']
    experiment = conf['name']
    id = conf['id']
    save = conf['save']
    verbose = conf['verbose']

    aug = functools.partial(utils.diff_augment, strategy=augmentations, param=utils.ParamDiffAug())
    # Load the dataset
    real_data = utils.get_dataset(dataset, data_path)
    torch.backends.cudnn.benchmark = True

        
    if mode == "Real":
        loader = utils.DeviceDataLoader(real_data['train'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, device=device)
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
    eval_data = utils.DeviceDataLoader(real_data['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)


    print(f"Training {mode} {model_name}")
    model = TimmModel(model_name, real_data['num_classes'], drop_rate=drop_rate, pretrained=pretrained).to(device)
    model.encoder_grad(not pretrained)
    model.head_grad(True)
    train_loader = utils.TransformLoader(loader, utils.get_transforms(model, mode, pretrained))
    eval_loader = utils.TransformLoader(eval_data, utils.get_transforms(model, 'Real', pretrained))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(loader)*num_epochs)
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
        model_path = checkpoint_path / f"{dataset}/{model.name}/pretrained_{pretrained}/{experiment}/{mode}_exp{exp}_acc{int(max(test_accs)*1000)}_id{id}.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "latest_state_dict": model.state_dict(),
            "best_state_dict": best_model,
            **conf,
            **result
        }, 
        model_path)
    return result


def objective(config, trial, device):
    cfg = {}
    use_strategy = False
    for name, value in config.items():
        if "@" in name:
            name = name.replace("@", "")
            value = utils.get_opuna_value(name, value, trial)
        elif name == "strategy":
            use_strategy = True
            res = {"name": value["name"], 'args':{}}
            for k, v in value["args"].items():
                if "@" in k:
                    k = k.replace("@", "")
                    _k = 'strategy.'+k
                    res['args'][k] = utils.get_opuna_value(_k, v, trial)
                else:
                    res['args'][k] = v
            value = res
        obj = utils.str_to_torch(name, value)
        cfg[name] = obj
    cfg['id'] = trial.number
    cfg['device'] = device
    iterations = cfg['iterations']
    evals = {}
    for i in range(iterations):
        print(f"Running trial {trial.number}, device {cfg['device']}, iteration {i+1}/{iterations}")
        if use_strategy:
            eval_data = train_strategy_conf(cfg)
        else:
            eval_data = train_baseline_conf(cfg)
        v = float(np.max(eval_data['test_acc']))
        evals["min"] = min(evals.get("min", 1), v)    
        evals["max"] = max(evals.get("max", 0), v)
        evals["sum"] = evals.get("sum", 0) + v
        evals["count"] = evals.get("count", 0) + 1


    avg = evals["sum"]/evals["count"]
    trial.set_user_attr('min', evals["min"])
    trial.set_user_attr('max', evals["max"])
    trial.set_user_attr('avg', avg)
    for k, v in cfg.items():
        trial.set_user_attr(k, v)
    return avg

def train_optuna():
    parser= argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    path = args.config
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = copy.deepcopy(config)
    jobs = []
    storage_name = config['db']
    print(f"Use: optuna-dashboard {storage_name}")
    names = []
    values = []
    for name, value in config.items():
        if "$" in name:
            _name = name.replace("$", "")
            random.shuffle(value)
            names.append(_name)
            values.append(value)
            cfg.pop(name)
    
    for vs in itertools.product(*values):
        study_name = f'{cfg["name"]}'
        for i, n in enumerate(names):
            cfg[n] = vs[i]
            if n == 'strategy':
                study_name += f'_{n}_{vs[i]["name"]}'
            else:
                study_name += f'_{n}_{vs[i]}'
        trials = cfg['trials']
        partial_objective = functools.partial(objective, copy.deepcopy(cfg))
        jobs += [(partial_objective, study_name, storage_name) for _ in range(trials)]

    gpu_nodes = cfg['nodes']*cfg['jobs_per_node']
    random.shuffle(jobs)
    utils.parallelize(utils.ask_tell_optuna, jobs, gpu_nodes, timeout=60*60*48)


def train_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        conf = yaml.safe_load(file)
    train_baseline_conf(conf)

if __name__ == "__main__":
    train_baseline()