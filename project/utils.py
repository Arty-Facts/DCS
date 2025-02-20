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
import project.models.TIMM as timm

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
                    print(f'Job on cpu:{worker.node} in slot {worker.id} timed out. Killing it...')
                worker.process.terminate()
            if not worker.process.is_alive():
                worker.process.kill()
                if verbose:
                    print(f'Launching job on cpu:{worker.node} in slot {worker.id}. {len(jobs)} jobs to left...')
                if len(jobs) == 0:
                    break
                args = list(jobs.pop())
                args.append(f'cpu:{worker.node}')
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

def get_strategy(strategy, generator, model, real_data, device):
    if strategy['name'] == 'EilertsenEscape':
        strat = strategies.EilertsenEscape(**strategy['strat_args'])
    elif strategy['name'] == 'Matcher':
        model_name = strategy['strat_args'].pop('model_name', None)
        if model_name is not None:
            strat = strategies.Matcher(model=model, **strategy['strat_args'])
        else:
            encoder_model = timm.TimmModel(model_name, 0,  pretrained=True).to(device)
            strat = strategies.Matcher(model=encoder_model, **strategy['strat_args'])
    elif strategy['name'] == 'RandomNoise':
        strat =  strategies.RandomNoise(**strategy['strat_args'])
    else:
        raise ValueError(f'Invalid strategy: {strategy}')
    return strategies.LatentWalker(generator, real_data, model, strategy=strat, **strategy['walker_args'])

def train(model, loader, optimizer, criterion, lr_scheduler=None, scaler=None, aug=None, transform=None, verbose=False):
    loss_sum = 0
    train_acc = 0
    model.train()
    if verbose:
        data_iter = tqdm.tqdm(loader, desc='train')
    else:
        data_iter = loader
    iter_count = 0
    for x, y, i in data_iter:
        if transform is not None:
            x = transform(x)
        if aug is not None:
            x = aug(x)
        optimizer.zero_grad()
        # with torch.cpu.amp.autocast(dtype=torch.bfloat16):
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
        iter_count += 1
    train_acc /= iter_count * loader.batch_size
    loss_sum /= iter_count
    return train_acc,  loss_sum

def test(model, loader, criterion, transform=None, verbose=False):
    loss_sum = 0
    test_acc = 0
    model.eval()
    if verbose:
        data_iter = tqdm.tqdm(loader, desc='test')
    else:
        data_iter = loader
    iter_count = 0
    with torch.no_grad():
        for x, y, i in data_iter:
            if transform is not None:
                x = transform(x)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_acc += (y_hat.argmax(-1) == y).float().sum().item()
            loss_sum += loss.item()
            iter_count += 1
    test_acc /= iter_count * loader.batch_size
    loss_sum /= iter_count
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
    def __setattr__(self, __name: str, __value) -> None:
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
        model.load_state_dict(torch.load(weight_path, weights_only=False))
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

def imge_stack(batch):
    data = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    index_tensors = [item[2] for item in batch]
    targets = torch.tensor(target_tensors, dtype=torch.long)
    index = torch.tensor(index_tensors, dtype=torch.long)
    return data, targets, index
class Composite(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self):
        return f'Composite({self.transforms})'
    
def get_interpolation_mode(transform):
    if isinstance(transform, transforms.Resize):
        mode = ""
        if transform.interpolation == torchvision.transforms.InterpolationMode.BILINEAR:
            mode = 'bilinear'
        elif transform.interpolation == torchvision.transforms.InterpolationMode.NEAREST:
            mode = 'nearest'
        elif transform.interpolation == torchvision.transforms.InterpolationMode.BICUBIC:
            mode = 'bicubic'
        else:
            raise ValueError(f'Invalid interpolation mode: {transform.interpolation}')
        return mode
    else:
        raise ValueError(f'Invalid transform: {transform}')
    

def get_transforms(model, mode, pretrained=True, device='cpu'):
    if pretrained:
        if mode == 'Real':
            return model.transform
        else:
            layers = []
            for t in model.transform.transforms:
                if isinstance(t, transforms.Resize):
                    mode = get_interpolation_mode(t)
                    layers.append(torch.nn.Upsample(size=t.size, mode=mode))
                elif isinstance(t, transforms.Normalize):
                    layers.append(t)
                elif isinstance(t, transforms.CenterCrop):
                    layers.append(t)
                else:
                    print(f'Warning: Ignoring transform {t}')
            return Composite(layers).to(device)
    else:
        if mode == 'Real':
            return torchvision.transforms.Compose([transforms.ToTensor()] + [t for t in model.transform.transforms if isinstance(t, transforms.Normalize)])
        else:
            layers = []
            for t in model.transform.transforms:
                if isinstance(t, transforms.Resize):
                    mode = get_interpolation_mode(t)
                    layers.append(torch.nn.Upsample(size=t.size, mode=mode))
                elif isinstance(t, transforms.Normalize):
                    layers.append(t)
                elif isinstance(t, transforms.CenterCrop):
                    layers.append(t)
                else:
                    print(f'Warning: Ignoring transform {t}')
            return Composite(layers).to(device)
    
def get_encoder_transform(model, device='cpu'):
    layers = []
    for t in model.transform.transforms:
        if isinstance(t, transforms.Resize):
            mode = get_interpolation_mode(t)
            layers.append(torch.nn.Upsample(size=t.size, mode=mode))
        elif isinstance(t, transforms.CenterCrop):
            layers.append(t)
        elif isinstance(t, transforms.Normalize):
            layers.append(t)
        else:
            print(f'Warning: Ignoring transform {t}')
    return Composite(layers).to(device)

class DatasetLoader():
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, collate_fn=diff_stack):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if drop_last:
            self.len = len(self.dataset)//self.batch_size
        elif len(self.dataset) % self.batch_size == 0:
            self.len = len(self.dataset)//self.batch_size
        else:
            self.len = len(self.dataset)//self.batch_size + 1
        self.collate_fn = collate_fn


    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))

        for i in range(0, len(indices), self.batch_size):
            batch = [self.dataset[j] for j in indices[i:i+self.batch_size]]
            yield self.collate_fn(batch)
            
        
    def __len__(self):
        return self.len

    def set_transform(self, transform):
        self.dataset.transform = transform

class TensorDataset():
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        return self.data[index], self.target[index], index
    def __len__(self):
        return len(self.data)

class GeneratorDatasetLoader():
    def __init__(self, anchors, labels, generator, config, weight_path, batch_size, shuffle=True, num_workers=0, device="cpu", use_cache=True, generator_grad=True, test=-1):
        if test != -1:
            labels = labels[:test]
            anchors = anchors[:test]
        self.labels = labels.long()
        if use_cache:
            self.anchors = anchors
        else:
            self.anchors = anchors.to(device)
        self.original_anchors = self.anchors.clone()
        self.generator_name = generator
        self.config = config
        self.dataset = TensorDataset(self.anchors, self.labels)
        self.generator, self.mean, self.std = load_generator(generator, config, weight_path)
        if not use_cache:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device
       
        self.loader = DatasetLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=diff_stack)
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
            print('Done!')
        self.generator.requires_grad_(generator_grad)
        if not self.use_cache:
            self.generator.to(self.device)

    def generate(self, data, label):
        imgs = generate(self.generator, data, label, self.mean, self.std)
        return imgs

    def __iter__(self):
        for batch in self.loader:
            data, label, index = batch
            label = label.to(self.device)
            if self.use_cache:
                imgs = self.cache[index]
            else:
                data = data.to(self.device)
                imgs = self.generate(data, label)

            yield imgs, label, index
    
    def __len__(self):
        return len(self.loader)
    
    def set_transform(self, transform):
        self.transform = transform

    def reset(self):
        self.anchors.data = self.original_anchors.data.clone()
    

                                        
    
class TransformLoader():
    def __init__(self, dataloader, transform, device="cpu"):
        self.dataloader = dataloader
        self.transform = transform
        self.batch_size = dataloader.batch_size
        self.device = device


    def __iter__(self):
        for batch in self.dataloader:
            data, target, *rest = batch
            if self.transform is None:
                data = torch.stack([d for d in data]).to(self.device)
            else:
                data = self.transform(torch.stack([d for d in data]).to(self.device))
            target = target.to(self.device)
            yield data, target, *rest
        
    def __len__(self):
        return len(self.dataloader)
    
class ImageLoader():
    def __init__(self, dataloader, transform, device="cpu"):
        self.dataloader = dataloader
        self.transform = transform
        self.batch_size = dataloader.batch_size
        self.device = device


    def __iter__(self):
        for batch in self.dataloader:
            data, target, *rest = batch
            if self.transform is None:
                data = torch.stack([d.to(self.device) for d in data])
            else:
                data = torch.stack([self.transform(d).to(self.device) for d in data])
            target = target.to(self.device)
            yield data, target, *rest
        
    def __len__(self):
        return len(self.dataloader)



def load_anchors(data_path):
    data = torch.load(data_path, weights_only=False)
    anchors = torch.from_numpy(data['anchors'])
    labels = torch.from_numpy(data['labels'])
    generator = data['generator']
    config = data['config']
    return anchors, labels, generator, config

def get_generator(anchors_path, weight_path, shuffle=True, num_workers=0, batch_size=64, device="cpu", use_cache=True, generator_grad=True):
    anchors_data = load_anchors(anchors_path)
    return GeneratorDatasetLoader(*anchors_data, weight_path, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,  device=device, use_cache=use_cache, generator_grad=generator_grad)

def generator_path(generator_path, dataset, exp):
    return f'{generator_path}_{dataset}_exp{exp}.pth'

def anchor_path(anchor_path, mode, dataset, exp):
    return f'{anchor_path}_{mode}_{dataset}_exp{exp}.pt'

