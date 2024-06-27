import torch
import torch.nn as nn
import tqdm
import project.utils as utils


class EilertsenEscape:
    def loss_up(self, xs, ys, gen_images, real_images):
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return -ce
    
    def loss_down(self, xs, ys, gen_images, real_images):
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return ce
    
    def scoring(self, xs, ys, gen_images, real_images):
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return ce

    
class Matcher():
    def __init__(self,
                 model,
                 margin=0.25):
        self.margin = margin
        self.model = model
     
    
    def loss_up(self, x, y, gen_images, real_images):
        emb_gen = self.model.encode(gen_images)
        emb_real = self.model.encode(real_images)
        simularity = max(nn.functional.cosine_similarity(emb_gen, emb_real).mean(), self.margin)
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        return -ce * simularity
    
    def loss_down(self, x, y, gen_images, real_images):
        emb_gen = self.model.encode(gen_images)
        emb_real = self.model.encode(real_images)
        simularity = max(1-nn.functional.cosine_similarity(emb_gen, emb_real).mean(), self.margin)
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        return ce * simularity 
    
    def scoring(self, x, y, gen_images, real_images):
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        return ce


class JonssonNoise():
    def __init__(self, step_size=0.01):
        self.step_size = step_size
        
    def loss_up(self, xs, ys, gen_images, real_images):
        rad = torch.randn_like(xs) * self.step_size
        xs = xs + rad
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return -ce 
    
    def loss_down(self, xs, ys, gen_images, real_images):
        rad = torch.randn_like(xs) * self.step_size
        xs = xs + rad
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return ce
    
    def scoring(self, xs, ys, gen_images, real_images):
        ce = nn.functional.cross_entropy(xs, ys, reduction='none')
        return ce

    
class LatentWalker():
    def __init__(self,
                 generator,
                 real_data,
                 model,
                 lr=0.01, weight_decay=0.0, max_iter=5,
                 upper_threshold=None, lower_threshold=None,
                 strategy=EilertsenEscape(),
                 cache_intermediate=False):
        self.generator = generator
        self.real_data = real_data.dataset
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.transform = utils.get_transforms(model, "GEN")
        self.real_transform = utils.get_transforms(model, "Real")
        self.intermediate = None
        self.cache_intermediate = cache_intermediate
        self.upper_threshold = upper_threshold if upper_threshold is not None and upper_threshold < 1 else None
        self.lower_threshold = lower_threshold if lower_threshold is not None and lower_threshold > 0 else None
        self.strategy = strategy
        if cache_intermediate:
            self.intermediate = [self.generator.anchors.detach().clone().cpu()]
    
    @torch.no_grad()  
    def rank(self, index_to_update, scores, anchors, labels, score_fn, batch_size, device):
        for index_batch in index_to_update.split(batch_size):
            x = torch.stack([anchors[index].to(device) for index in index_batch])
            y = torch.stack([labels[index].to(device) for index in index_batch])
            images = self.generator.generate(x, y)
            images = torch.stack([self.transform(image) for image in images])
            real_images = torch.stack([self.real_transform(self.real_data[i][0]) for i in index_batch]).to(x.device)
            logits = self.model(images)
            batch_score = score_fn(logits, y, images, real_images)
            for index, score in zip(index_batch, batch_score):
                scores[index] = score.item()
        return scores
    
    def update(self, index_to_update, anchors, labels, losses, optimizers, loss_func, batch_size, device):
        shuffled_index = torch.randperm(len(index_to_update))
        for index_batch in shuffled_index.split(batch_size):
            index_batch = index_to_update[index_batch]
            for index in index_batch:
                optimizers[index].zero_grad()
            x = torch.stack([anchors[index].to(device) for index in index_batch])
            y = torch.stack([labels[index].to(device) for index in index_batch])
            images = self.generator.generate(x, y)
            images = torch.stack([self.transform(image) for image in images])
            real_images = torch.stack([self.real_transform(self.real_data[i][0]) for i in index_batch]).to(x.device)
            logits = self.model(images)
            batch_loss = loss_func(logits, y, images, real_images)
            tot_loss = batch_loss.sum()
            tot_loss.backward()
            for index, loss in zip(index_batch, batch_loss):
                optimizers[index].step()
                losses[index] = loss.item()

                
        
    def __call__(self,  verbose=False):
        self.generator.reset()
        learnable = [a for a in self.generator.anchors]
        for p in learnable:
            p.requires_grad = True
        labels = self.generator.labels
        device = self.generator.device
        batch_size = self.generator.batch_size
        if verbose:
            display = tqdm.tqdm(total=self.max_iter, desc=self.__class__.__name__)

        scores = torch.zeros(len(labels)) 
        all_index = torch.arange(len(labels))
        if verbose:
            display.set_description(f"Ranking") 
        self.rank(all_index, scores, learnable, labels, self.strategy.scoring, batch_size, device)
        sorted_score = scores.clone().sort().values
        
        optimizers = [torch.optim.Adam([p], lr=self.lr, weight_decay=self.weight_decay) for p in learnable]
        
        losses = torch.zeros(len(labels))
        push_up = torch.zeros(len(labels), dtype=torch.bool)
        push_down = torch.zeros(len(labels), dtype=torch.bool)
        if self.upper_threshold is None:
            upper_score = None
        else:
            upper_score = sorted_score[int(len(sorted_score)*self.upper_threshold)]
        if self.lower_threshold is None:
            lower_score = None
        else:
            lower_score = sorted_score[int(len(sorted_score)*self.lower_threshold)]
        self.saved_scores = [sorted_score]
        self.saved_up = [(all_index[push_up].clone(), 
                          scores[push_up].clone(), 
                          self.generator.anchors[push_up].clone().cpu())]
        self.saved_down = [(all_index[push_down].clone(), 
                            scores[push_down].clone(), 
                            self.generator.anchors[push_down].clone().cpu())]
        
        iter_count = 0
        while True:
            push_up.fill_(False)
            push_down.fill_(False)
            if lower_score is not None:
                push_up[scores < lower_score] = True
            if upper_score is not None:
                push_down[scores > upper_score] = True
            if push_up.sum() == 0 and push_down.sum() == 0:
                break
            if verbose:
                display.set_description(f"Push Up: {push_up.sum()}, Push Down: {push_down.sum()}")
            iter_count += 1
            
            if lower_score is not None:
                self.update(all_index[push_up], learnable, labels, losses, optimizers, self.strategy.loss_up, batch_size, device)
            if upper_score is not None:
                self.update(all_index[push_down], learnable, labels, losses, optimizers, self.strategy.loss_down, batch_size, device)
            if verbose:
                display.set_description(f"Ranking")
            self.rank(all_index[push_down | push_up], scores, learnable, labels, self.strategy.scoring, batch_size, device)
     
            
            self.saved_scores.append(scores.clone().sort().values)
            if lower_score is not None:
                self.saved_up.append((all_index[push_up].clone(), 
                                    scores[push_up].clone(), 
                                    self.generator.anchors[push_up].clone().cpu()))
            if upper_score is not None:
                self.saved_down.append((all_index[push_down].clone(), 
                                        scores[push_down].clone(), 
                                        self.generator.anchors[push_down].clone().cpu()))
        
            if verbose:
                display.update(1)
            
            if iter_count == self.max_iter:
                break

        if self.cache_intermediate:
            self.intermediate.append(self.generator.anchors.detach().clone().cpu())
        if verbose:
            display.close()
        
    


    
