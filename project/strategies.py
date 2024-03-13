import torch
import torch.nn as nn
import tqdm

class Static():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

class EilertsenEscape():
    def __init__(self,
                 generator,
                 model,
                 lr=0.01, weight_decay=0.0, 
                 reset_every=10,
                 cache_intermediate=False,
                 criterion_classifier=nn.CrossEntropyLoss()):
        learnable = [generator.anchors]
        for p in learnable:
            p.requires_grad = True
        self.learnable = learnable
        self.generator = generator
        self.model = model
        self.original = [p.detach().clone() for p in learnable]
        self.optimizer = torch.optim.SGD(learnable, lr=lr, weight_decay=weight_decay)
        self.criterion = criterion_classifier
        self.reset_every = reset_every
        self.epoch_counter = 0
        self.intermediate = None
        self.cache_intermediate = cache_intermediate
        self.reset_counter = 0
        if cache_intermediate:
            self.intermediate = [[self.generator.anchors.detach().clone().cpu()]]

    def __call__(self,  verbose=False):
        if self.epoch_counter % self.reset_every == 0 and self.epoch_counter > 0:
            self.reset()
        self.epoch_counter += 1
        if verbose:
            data_iter = tqdm.tqdm(self.generator, desc=self.__class__.__name__)
        else:
            data_iter = self.generator
        self.optimizer.zero_grad()
        for x, y, i in data_iter:
            y_hat = self.model(x)
            loss =  -self.criterion(y_hat, y)
            loss.backward()

        self.optimizer.step()
        if self.cache_intermediate:
            self.intermediate[self.reset_counter].append(self.generator.anchors.detach().clone().cpu())
        return self
    
    def reset(self):
        for i, p in enumerate(self.learnable):
            p.data = self.original[i].clone()
        self.reset_counter += 1
        if self.cache_intermediate:
            self.intermediate.append([self.generator.anchors.detach().clone().cpu()])
    
class LPInversion():
    def __init__(self,
                 generator,
                 model,
                 real_data,
                 lr=0.01, weight_decay=0.0, alpha=0.5,
                 cache_intermediate=False,
                 criterion_classifier=nn.MSELoss()):
        learnable = [generator.anchors]
        for p in learnable:
            p.requires_grad = True
        self.learnable = learnable
        self.generator = generator
        self.real_data = real_data.dataset
        self.model = model
        self.original = [p.detach().clone() for p in learnable]
        self.optimizer = torch.optim.Adam(learnable, lr=lr, weight_decay=weight_decay)
        self.criterion = criterion_classifier
        self.intermediate = None
        self.alpha = alpha
        self.cache_intermediate = cache_intermediate
        self.reset_counter = 0
        if cache_intermediate:
            self.intermediate = [[self.generator.anchors.detach().clone().cpu()]]

    def __call__(self,  verbose=False):
        if verbose:
            data_iter = tqdm.tqdm(self.generator, desc=self.__class__.__name__)
        else:
            data_iter = self.generator
        self.optimizer.zero_grad()
        
        for x, y, i in data_iter:
            real_x = torch.stack([self.real_data[p][0] for p in i]).to(x.device)
            emb = self.model.encode(x)
            real_emb = self.model.encode(real_x)
            loss = self.alpha*self.criterion(emb, real_emb) + (1-self.alpha)*self.criterion(x, real_x)
            loss.backward()

        self.optimizer.step()
        if self.cache_intermediate:
            self.intermediate[0].append(self.generator.anchors.detach().clone().cpu())
        return self