import torch
import numpy as np
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import time
import multiprocessing as mp
import random
import optuna
import tqdm
from project.BigGAN import Generator
import project.strategies as strategies

class dummy:
    @staticmethod
    def is_alive():
        return False
    @staticmethod
    def close():
        pass
    @staticmethod
    def terminate():
        pass
    @staticmethod
    def join():
        pass
    @staticmethod
    def kill():
        pass

class Worker:
    def __init__(self, id, node):
        self.id = id
        self.node = node
        self.process = dummy
        self.start_time = time.perf_counter()


def parallelize(func, jobs, gpu_nodes, verbose=True, timeout=60*60*24):
    if verbose:
        print(f'Launching {len(jobs)} jobs on {len(set(gpu_nodes))} GPUs. {len(gpu_nodes)//len(set(gpu_nodes))} jobs per GPU in parallel..')
    workers = [Worker(id, node) for id, node in enumerate(gpu_nodes)]
    while len(jobs) > 0:
        random.shuffle(workers)
        for worker in workers:
            if time.perf_counter() - worker.start_time > timeout:
                if verbose:
                    print(f'Job on cuda:{worker.node} in slot {worker.id} timed out. Killing it...')
                worker.process.terminate()
            if not worker.process.is_alive():
                worker.process.kill()
                if verbose:
                    print(f'Launching job on cuda:{worker.node} in slot {worker.id}. {len(jobs)} jobs to left...')
                if len(jobs) == 0:
                    break
                args = list(jobs.pop())
                args.append(f'cuda:{worker.node}')
                p = mp.Process(target=func, args=args)
                p.start()
                worker.process = p
                worker.start_time = time.perf_counter()
                time.sleep(1)
        time.sleep(1)
    for worker in workers:
        worker.process.join()
    if verbose:
        print('Done!')


def ask_tell_optuna(objective_func, study_name, storage_name, device):
    study = optuna.create_study(directions=['maximize'], study_name=study_name, storage=storage_name, load_if_exists=True)
    trial = study.ask()
    res = objective_func(trial, device)
    study.tell(trial, res)


def get_opuna_value(name, opt_values, trial):
    data_type,*values = opt_values
    if data_type == "int":
        min_value, max_value, step_scale = values
        if step_scale == "log":
            return trial.suggest_int(name, min_value, max_value, log=True)
        elif step_scale.startswith("uniform_"):
            step = int(step_scale.split("_")[1])
            return trial.suggest_int(name, min_value, max_value, step=step)
        else:
            return trial.suggest_int(name, min_value, max_value)
    elif data_type == "float":
        min_value, max_value, step_scale = values
        if step_scale == "log":
            return trial.suggest_float(name, min_value, max_value, log=True)
        elif step_scale.startswith("uniform_"):
            step = float(step_scale.split("_")[1])
            return trial.suggest_float(name, min_value, max_value, step=step)
        else:
            return trial.suggest_float(name, min_value, max_value)
    elif data_type == "categorical":
        return trial.suggest_categorical(name, values[0])
    else:
        raise ValueError(f"Unknown data type {data_type}")
    
def str_to_torch(name, value):
    return value

def get_strategy(strategy, generator, model, real_data):
    if strategy['name'] == 'Static':
        return strategies.Static()
    elif strategy['name'] == 'EilertsenEscape':
        return strategies.EilertsenEscape(generator, model, **strategy['args'])
    elif strategy['name'] == 'LatentInversion':
        return strategies.LatentInversion(generator, model, real_data, **strategy['args'])
    elif strategy['name'] == 'PixelInversion':
        return strategies.PixelInversion(generator, real_data, **strategy['args'])
    elif strategy['name'] == 'LPInversion':
        return strategies.LPInversion(generator, model, real_data, **strategy['args'])
    else:
        raise ValueError(f'Invalid strategy: {strategy}')

def train(model, loader, optimizer, criterion, lr_scheduler=None, scaler=None, aug=None, verbose=False):
    loss_sum = 0
    train_acc = 0
    model.train()
    if verbose:
        data_iter = tqdm.tqdm(loader, desc='train')
    else:
        data_iter = loader
    for x, y, i in data_iter:
        if aug is not None:
            x = aug(x)
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        train_acc += (y_hat.argmax(-1) == y).float().sum().item()
        loss_sum += loss.item()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    train_acc /= len(loader.dataset)
    loss_sum /= len(loader)
    return train_acc,  loss_sum

def test(model, loader, criterion, verbose=False):
    loss_sum = 0
    test_acc = 0
    model.eval()
    if verbose:
        data_iter = tqdm.tqdm(loader, desc='test')
    else:
        data_iter = loader
    with torch.no_grad():
        for x, y, i in data_iter:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_acc += (y_hat.argmax(-1) == y).float().sum().item()
            loss_sum += loss.item()
    test_acc /= len(loader.dataset)
    loss_sum /= len(loader)
    return test_acc, loss_sum

def renormalize(img, mean, std):
    mean = mean.view(1, -1, 1, 1).to(img.device)
    std = std.view(1, -1, 1, 1).to(img.device)
    return img * std + mean

def generate(model, z, lab, mean, std):
    num_max = 500  # Error occurs when batch size of G is large.
    num = z.shape[0]
    if num > num_max:
        img_syn = []
        for i in range(int(np.ceil(num / num_max))):
            img_syn.append(renormalize(model(z[i * num_max: (i + 1) * num_max], lab[i * num_max: (i + 1) * num_max]), mean, std))
        return torch.cat(img_syn, dim=0)
    else:
        return renormalize(model(z, lab), mean, std)

def get_label(dataset):
    if isinstance(dataset, (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10, datasets.CIFAR100)):
        return torch.tensor(dataset.targets)
    elif isinstance(dataset, datasets.SVHN):
        return torch.tensor(dataset.labels)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

class IndexDataset():
    def __init__(self, dataset):
        self.dataset = dataset
    def __setattr__(self, __name: str, __value: torch.Any) -> None:
        if 'dataset' in self.__dict__:
            setattr(self.dataset, __name, __value)
        elif __name == 'dataset':
            self.__dict__[__name] = __value
    def __getitem__(self, index):
        return *self.dataset[index], index
    def __len__(self):
        return len(self.dataset)

def get_dataset(dataset, data_path, transform=None):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    dst_test = IndexDataset(dst_test)
    dst_train = IndexDataset(dst_train)
    return {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'train': dst_train,
        'test': dst_test,
        'label_train': get_label(dst_train.dataset),
        'label_test': get_label(dst_test.dataset),
        'class_names': class_names
    }


def distance_wb(gwr, gws):
    shape = gwr.shape

    if len(shape) > 2:
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, metric, device):
    if metric == 'itgan':
        dis = torch.tensor(0.0, device=device)
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    else:
        raise ValueError(f'Invalid metric: {metric}')
    return dis

def load_generator(generator, config, weight_path):
    if generator == 'BigGAN':
        model = Generator(**config)
        model.load_state_dict(torch.load(weight_path))
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    else:
        raise ValueError(f'Invalid generator: {generator}')
    return model, mean, std

def diff_stack(batch):
    data_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    index_tensors = [item[2] for item in batch]

    # Stack data tensors and target tensors separately to allow gradient computation
    data = torch.stack(data_tensors)
    targets = torch.stack(target_tensors)
    index = torch.stack(index_tensors)
    return data, targets, index

def set_transforms(model, mode, loader, eval, pretrained=True):
    if pretrained:
        if eval is not None:
            eval.set_transform(model.transform)
        if mode == 'Real':
            loader.set_transform(torchvision.transforms.Compose(model.transform.transforms))
        else:
            loader.set_transform(torchvision.transforms.Compose([t for t in model.transform.transforms if not isinstance(t, transforms.ToTensor)])) 
    else:
        if eval is not None:
            eval.set_transform(torchvision.transforms.Compose([transforms.ToTensor()] + [t for t in model.transform.transforms if isinstance(t, transforms.Normalize)]))
        if mode == 'Real':
            loader.set_transform(torchvision.transforms.Compose([transforms.ToTensor()] + [t for t in model.transform.transforms if isinstance(t, transforms.Normalize)]))
        else:
            loader.set_transform(torchvision.transforms.Compose([t for t in model.transform.transforms if isinstance(t, transforms.Normalize)]))
            # loader.set_transform(None)

class GeneratorDatasetLoader():
    def __init__(self, anchors, labels, generator, config, weight_path, batch_size, shuffle=True, num_workers=0, device="cuda", use_cache=True, generator_grad=True):
        self.labels = labels.long()
        self.anchors = anchors
        self.generator_name = generator
        self.config = config
        self.dataset = torch.utils.data.TensorDataset(self.anchors, self.labels, torch.arange(len(self.labels)))
        self.generator, self.mean, self.std = load_generator(generator, config, weight_path)
        self.mean = self.mean
        self.std = self.std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device
       
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=diff_stack)
        self.transform = None
        self.generator.eval()
        self.use_cache = use_cache

        if self.use_cache:
            self.generator.requires_grad_(False)
            print('Caching generator outputs...')
            self.generator.to(self.device)
            self.cache = [None for _ in range(len(self.labels))]
            for batch in self.loader:
                data, label, index = batch
                data = data.to(self.device)
                label = label.to(self.device)
                imgs = generate(self.generator, data, label, self.mean, self.std)
                for i, img in zip(index, imgs):
                    self.cache[i] = img
            self.cache = torch.stack(self.cache, dim=0)
            self.generator.to('cpu')
            print('Done!')
        self.generator.requires_grad_(generator_grad)

    def __iter__(self):
        if not self.use_cache:
            self.generator.to(self.device)
        for batch in self.loader:
            data, label, index = batch
            label = label.to(self.device)
            if self.use_cache:
                imgs = self.cache[index]
            else:
                data = data.to(self.device)
                imgs = generate(self.generator, data, label, self.mean, self.std)
        
            # imgs = self.generator(data, label)
            if self.transform:
                imgs = torch.stack([self.transform(img) for img in imgs])
            yield imgs, label, index
        if not self.use_cache:
            self.generator.to('cpu')
    
    def __len__(self):
        return len(self.loader)
    
    def set_transform(self, transform):
        self.transform = transform

    def reset(self):
        self.anchors.data = self.original_anchors.data.clone()


class DeviceDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, device="cuda"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def __iter__(self):
        for batch in self.loader:
            data, label, index = batch
            data = data.to(self.device)
            label = label.to(self.device)
            yield data, label, index
        
    def __len__(self):
        return len(self.loader)

    def set_transform(self, transform):
        self.dataset.transform = transform


def load_anchors(data_path):
    data = torch.load(data_path)
    anchors = torch.from_numpy(data['anchors'])
    labels = torch.from_numpy(data['labels'])
    generator = data['generator']
    config = data['config']
    return anchors, labels, generator, config

def get_generator(anchors_path, weight_path, shuffle=True, num_workers=0, batch_size=64, device="cuda", use_cache=True, generator_grad=True):
    anchors_data = load_anchors(anchors_path)
    return GeneratorDatasetLoader(*anchors_data, weight_path, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,  device=device, use_cache=use_cache, generator_grad=generator_grad)

def generator_path(generator_path, dataset, exp):
    return f'{generator_path}_{dataset}_exp{exp}.pth'

def anchor_path(anchor_path, mode, dataset, exp):
    return f'{anchor_path}_{mode}_{dataset}_exp{exp}.pt'

