import ood_detectors.vision as vision_ood
import torch
import pathlib
import project.utils as utils
from collections import defaultdict
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ood_detectors.likelihood as likelihood
import ood_detectors.residual as residual
import seaborn as sns
from baseline_model import get_baseline_model
import yaml
import project.augmentations as aug_lib
import functools
from project.train_logger import TrainingLogger
import torchvision

PREFIX = "gan_sampler"

class CombineDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.batch_size = loaders[0].batch_size

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        self.at_loader_index = 0
        return self

    def __next__(self):
        next_data =  next(self.iterators[self.at_loader_index])
        self.at_loader_index = (self.at_loader_index + 1) % len(self.loaders)
        return next_data


def plot(id_score, gen_score, oods, dataset, tile, out_dir='figs', verbose=True, names=None, extra=None):
    if verbose:
        print('Generating plots...')


    # Create a figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))  # Adjust the size as needed
    fig.suptitle(f'{tile} Evaluation')

    def add_shadow(ax, data):
        if data.var() > 1e-6:
            l = ax.lines[-1]
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x,y, alpha=0.1)
            # Calculate and plot the mean
            mean_value = np.mean(data)
            line_color = l.get_color()
            ax.axvline(mean_value, color=line_color, linestyle=':', linewidth=1.5)

    # Subplot 1: KDE plots
    sns.kdeplot(data=id_score, bw_adjust=.2, ax=ax, label=f'Real training: {np.mean(id_score):.2f}')
    add_shadow(ax, id_score)

    sns.kdeplot(data=gen_score, bw_adjust=.2, ax=ax, label=f'Generated: {np.mean(gen_score):.2f}')
    add_shadow(ax, gen_score)

    if names is None:
        it = enumerate(oods)
    else:
        it = zip(names, oods)
    for index, score_ood in it:
        sns.kdeplot(data=score_ood, bw_adjust=.2, ax=ax, label=f'{index}: {np.mean(score_ood):.2f}')
        add_shadow(ax, score_ood)

    if extra is not None:
        for name, value in extra:
            ax.axvline(value, color='black', linestyle=':', linewidth=2.5)
            ax.text(value, 0, name, rotation=90, verticalalignment='bottom', horizontalalignment='center')

    ax.set_title('Density Plots')
    ax.set_xlabel('bits/dim')
    ax.set_ylabel('Density')

    ax.legend()


    # Save the figure
    out_dir = pathlib.Path(out_dir) / dataset
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{tile}.svg"
    # print(f"Saving figure to {out_dir / filename}")
    plt.savefig(out_dir / filename, bbox_inches='tight')

def unnormalize(x):
    return x.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def resize(x, size):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False).squeeze(0)
    else:
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

def main(config=None):
    if config is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose = True
    else:
        device = config['device']
        verbose = config['verbose']
    mode = 'ITGAN'
    dataset = 'CIFAR10'
    exp = 0
    batch_size = 128
    num_workers = 0
    clean = False

    root_path = pathlib.Path('/mnt/data/arty/data/IT-GAN')
    data_path = root_path / 'data'
    model_path = root_path / 'checkpoints'
    checkpoint_path = pathlib.Path('checkpoints')
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    figure_path = pathlib.Path('figs')/dataset
    figure_path.mkdir(exist_ok=True, parents=True)
    weight_path = model_path / f"G_Pretrained_{dataset}_exp{exp}.pth"

    encoder = vision_ood.get_encoder("dinov2")
    embedding_size = 768
    encoder.eval()
    encoder.to(device)
    encoder_name = encoder.name

    data = utils.get_dataset(dataset, data_path, None)
    anchors = utils.load_anchors(model_path / f'Base_ITGAN_{dataset}_exp{exp}.pt')
    train_img = data["train"]
    anchor_emb = anchors[0]
    label = anchors[1]
    generator_name = anchors[2]
    config = anchors[3]

    data_blob = defaultdict(list)

    baseline_model_config_path = 'project/config_baseline.yaml'
    with open(baseline_model_config_path, 'r') as f:
        baseline_model_config = yaml.safe_load(f)


    real_emb = torch.zeros((10, 5000, embedding_size))
    gen_emb = torch.zeros((10, 5000, embedding_size))
    encoder_transform = utils.get_encoder_transform(encoder, device=device)


    real_images = [[None]*5000 for _ in range(10)]
    gen_images = [[None]*5000 for _ in range(10)]
    for img, l, i in train_img:
        emb = anchor_emb[i]
        data_blob[int(l)].append((img, l, emb))
    if pathlib.Path(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and pathlib.Path(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and not clean:
        if verbose:
            print("Loading real embeddings")
        real_emb = torch.load(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
        real_images = torch.load(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
    else:
        if verbose:
            print("Calculating real embeddings")
        for l, v in tqdm.tqdm(data_blob.items()):
            images, labels, embeddings = zip(*v)
            indexs = list(range(len(images)))
            embeddings = torch.stack(embeddings)
            images = list(zip(images, labels, indexs))
            labels = torch.tensor(labels)
            real_loader = utils.ImageLoader(torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack), transform=torchvision.transforms.ToTensor(), device=device)
            with torch.no_grad():
                for (real_img, li, index) in real_loader:
                    for idx, ri in zip(index, real_img):
                        real_images[int(l)][idx] = ri.detach().cpu()
                    real_img = encoder_transform(real_img)
                    real_emb[li, index] = encoder(real_img).detach().cpu()
        with open(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(real_emb, f)
        with open(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(real_images, f)

    if pathlib.Path(checkpoint_path/f"gen_emb_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and pathlib.Path(checkpoint_path/f"gen_images_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and not clean:
        if verbose:
            print("Loading generated embeddings")
        gen_emb = torch.load(checkpoint_path/f"gen_emb_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
        gen_images = torch.load(checkpoint_path/f"gen_images_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
    else:
        if verbose:
            print("Calculating generated embeddings")
        for l, v in tqdm.tqdm(data_blob.items()):
            images, labels, embeddings = zip(*v)
            indexs = list(range(len(images)))
            embeddings = torch.stack(embeddings)
            images = list(zip(images, labels, indexs))
            labels = torch.tensor(labels)
            gen_loader = utils.TransformLoader(utils.GeneratorDatasetLoader(embeddings, labels, generator_name, config, weight_path, shuffle=False, num_workers=num_workers, batch_size=batch_size, device=device, use_cache=False), transform=None, device=device)
            with torch.no_grad():
                for (gen_img, li, index) in gen_loader:
                    for idx, gi in zip(index, gen_img):
                        gen_images[int(l)][idx] = gi.detach().cpu()
                    gen_img = encoder_transform(gen_img)
                    gen_emb[li, index] = encoder(gen_img).detach().cpu()
        with open(checkpoint_path/f"gen_emb_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(gen_emb, f)
        with open(checkpoint_path/f"gen_images_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(gen_images, f)

    pca = PCA(n_components=2)
    all_data = torch.cat([real_emb, gen_emb], dim=1)
    all_label = torch.cat([torch.arange(10), torch.arange(10)], dim=0)
    pca.fit(all_data.view(-1, embedding_size).numpy())
    pca_real = pca.transform(real_emb.view(-1, embedding_size).numpy())
    pca_gen = pca.transform(gen_emb.view(-1, embedding_size).numpy())
    fig, ax = plt.subplots(figsize=(20, 20))
    colors_hex = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00', '#FFA500', '#800080', '#FFC0CB', '#008000']
    for i in range(10):
        pca_real_i = pca_real[i*5000:(i+1)*5000]#.mean(axis=0).reshape(1, -1)
        pca_gen_i = pca_gen[i*5000:(i+1)*5000]#.mean(axis=0).reshape(1, -1)
        ax.scatter(pca_real_i[:, 0], pca_real_i[:, 1], label=f"Real {i}", color=colors_hex[i], alpha=0.5, marker='x')
        ax.scatter(pca_gen_i[:, 0], pca_gen_i[:, 1], label=f"Gen {i}", color=colors_hex[i], alpha=0.5, marker='o')

    ax.legend()
    plt.savefig(f"pca_{dataset}_{encoder_name}_{generator_name}.png")

    # save a plot if real and gen images
    fig, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        ax[0, i].imshow(real_images[i][0].permute(1, 2, 0))
        ax[0, i].axis('off')
        ax[1, i].imshow(gen_images[i][0].permute(1, 2, 0))
        ax[1, i].axis('off')
    plt.savefig(f"real_gen_{dataset}_{encoder_name}_{generator_name}.png")

    # ood_detectors = [likelihood.RDM(embedding_size).to("cpu") for _ in range(10)]
    ood_detectors = [residual.Residual(0.3) for _ in range(10)]
    if verbose:
        print("Fitting OOD Detectors")
    for i, ood_detector in tqdm.tqdm(list(enumerate(ood_detectors))):
        if pathlib.Path(checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt").exists() and not clean:
            ood_detector.load_state_dict(torch.load(checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt", weights_only=False))
            if verbose:
                print(f"OOD Detector {i} Loaded")
        else:
            ood_detector.to(device)
            ood_detector.fit(real_emb[i], batch_size=1000, n_epochs=1000, verbose=False)
            ood_detector.to("cpu")
            torch.save(ood_detector.state_dict(), checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt")
            if verbose:
                print(f"OOD Detector {i} Fitted")
    scores_real = torch.zeros((10, 10, 5000))
    scores_gen = torch.zeros((10, 10, 5000))
    if verbose:
        print("Calculating Scores")
    for i, ood_detector in tqdm.tqdm(list(enumerate(ood_detectors))):
        for index in range(10):
            if pathlib.Path(checkpoint_path / f'score_real_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt').exists() and not clean:
                scores_real[i][index] = torch.load(checkpoint_path / f'score_real_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt', weights_only=False)
            else:
                ood_detector.to(device)
                score_real = ood_detector.predict(real_emb[index], batch_size=1000, verbose=False)
                ood_detector.to("cpu")
                score_real = torch.from_numpy(score_real)
                torch.save(score_real, checkpoint_path / f'score_real_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt')
                scores_real[i][index] = score_real
            if pathlib.Path(checkpoint_path / f'score_gen_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt').exists() and not clean:
                scores_gen[i][index] = torch.load(checkpoint_path / f'score_gen_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt', weights_only=False)
            else:
                ood_detector.to(device)
                score_gen = ood_detector.predict(gen_emb[index], batch_size=1000, verbose=False)
                score_gen = torch.from_numpy(score_gen)
                ood_detector.to("cpu")
                torch.save(score_gen, checkpoint_path / f'score_gen_{ood_detector.name}_{i}_{index}_{dataset}_{encoder_name}.pt')
                scores_gen[i][index] = score_gen
    if verbose:
        print("Tuning Anchors")
    aug = functools.partial(aug_lib.diff_augment, strategy="color_crop_cutout_flip_scale_rotate", param=aug_lib.ParamDiffAug())
    eval_data = torch.utils.data.DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack)
    eval_loader = utils.ImageLoader(eval_data, transform=torchvision.transforms.ToTensor(), device=device)
    train_data = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.imge_stack)
    train_loader = utils.ImageLoader(train_data, transform=torchvision.transforms.ToTensor(), device=device)
    generator = utils.GeneratorDatasetLoader(*utils.load_anchors(model_path / f'Base_ITGAN_{dataset}_exp{exp}.pt'), weight_path, shuffle=True, num_workers=num_workers, batch_size=batch_size,  device=device, use_cache=True)
    dynamic_generator = utils.GeneratorDatasetLoader(*utils.load_anchors(model_path / f'Base_ITGAN_{dataset}_exp{exp}.pt'), weight_path, shuffle=True, num_workers=num_workers, batch_size=batch_size,  device=device, use_cache=False)


    all_emb = []
    all_images = []
    all_labels = []
    for i in range(10):
        img, label, emb = zip(*data_blob[i])
        all_emb.append(torch.stack(emb).clone())
        all_images.append(img)
        all_labels.append(torch.tensor(label))



    num_epochs = 100
    lr = 1e-4
    logger = TrainingLogger(f"{PREFIX}.db")

    exp_id_contolle = logger.register_experiment(name="Gen")
    exp_id_our = logger.register_experiment(name="Our")
    exp_id_real = logger.register_experiment(name="Real")
    exp_id_gen_real = logger.register_experiment(name="Gen+Real")
    exp_id_our_gen = logger.register_experiment(name="Our+Gen")
    exp_id_our_real = logger.register_experiment(name="Our+Real")
    exp_id_real_real = logger.register_experiment(name="Realx2")


    run_id_contolle = logger.get_next_run_id(exp_id_contolle)
    run_id_our = logger.get_next_run_id(exp_id_our)
    run_id_real = logger.get_next_run_id(exp_id_real)
    run_id_gen_real = logger.get_next_run_id(exp_id_gen_real)
    run_id_our_gen = logger.get_next_run_id(exp_id_our_gen)
    run_id_our_real = logger.get_next_run_id(exp_id_our_real)
    run_id_real_real = logger.get_next_run_id(exp_id_real_real)
    
    our_model = get_baseline_model(baseline_model_config, pre_trained=False)
    our_model.to('cpu')

    controlle_model = get_baseline_model(baseline_model_config, pre_trained=False)
    controlle_model.to('cpu')
    controlle_model.load_state_dict(our_model.state_dict())

    real_model = get_baseline_model(baseline_model_config, pre_trained=False)
    real_model.to('cpu')
    real_model.load_state_dict(our_model.state_dict())

    gen_real_model = get_baseline_model(baseline_model_config, pre_trained=False)
    gen_real_model.to('cpu')
    gen_real_model.load_state_dict(our_model.state_dict())

    our_gen_model = get_baseline_model(baseline_model_config, pre_trained=False)
    our_gen_model.to('cpu')
    our_gen_model.load_state_dict(our_model.state_dict())

    our_real_model = get_baseline_model(baseline_model_config, pre_trained=False)
    our_real_model.to('cpu')
    our_real_model.load_state_dict(our_model.state_dict())

    real_real_model = get_baseline_model(baseline_model_config, pre_trained=False)
    real_real_model.to('cpu')
    real_real_model.load_state_dict(our_model.state_dict())

    model_transform = utils.get_encoder_transform(our_model, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    controlle_optimizer = torch.optim.Adam(controlle_model.parameters(), lr=lr)

    our_optmizer = torch.optim.Adam(our_model.parameters(), lr=lr)

    real_optmizer = torch.optim.Adam(real_model.parameters(), lr=lr)

    gen_real_optmizer = torch.optim.Adam(gen_real_model.parameters(), lr=lr)

    our_gen_optmizer = torch.optim.Adam(our_gen_model.parameters(), lr=lr)

    our_real_optmizer = torch.optim.Adam(our_real_model.parameters(), lr=lr)

    real_real_optmizer = torch.optim.Adam(real_real_model.parameters(), lr=lr)

    experiments = [
        (controlle_model, controlle_optimizer, exp_id_contolle, run_id_contolle, generator, "Controlle"),
        (our_model, our_optmizer, exp_id_our, run_id_our, dynamic_generator, "Our"),
        (real_model, real_optmizer, exp_id_real, run_id_real, train_loader, "Real"),
        (gen_real_model, gen_real_optmizer, exp_id_gen_real, run_id_gen_real, CombineDataLoader([generator, train_loader]), "Gen+Real"),
        (our_gen_model, our_gen_optmizer, exp_id_our_gen, run_id_our_gen, CombineDataLoader([generator, dynamic_generator]), "Our+Gen"),
        (our_real_model, our_real_optmizer, exp_id_our_real, run_id_our_real, CombineDataLoader([dynamic_generator, train_loader]), "Our+Real"),
        (real_real_model, real_real_optmizer, exp_id_real_real, run_id_real_real, CombineDataLoader([train_loader, train_loader]), "Realx2"),
    ]
    if verbose:
        epoch_iter = tqdm.trange(num_epochs, desc=mode)
    else:
        epoch_iter = range(num_epochs)
    for epoch in epoch_iter:

        for model, optimizer, exp_id, run_id, loader, name in experiments:
            model.to(device)
            if verbose:
                epoch_iter.set_description(f"Epoch {epoch} - training {name}")
            model.train()
            train_acc, train_loss = utils.train(model, loader, optimizer, criterion, aug=aug, transform=model_transform)
            if verbose:
                epoch_iter.set_description(f"Epoch {epoch} - testing {name}")
            model.eval()
            test_acc, test_loss = utils.test(model, eval_loader, criterion, transform=model_transform)

            logger.report_result(exp_id, run_id, epoch, train_loss, train_acc, test_loss, test_acc)
            model.to('cpu')
        if verbose:
            epoch_iter.set_description(f"updating anchors")
        for i in range(10):
            samples = 10
            sorted_index_real = torch.argsort(scores_real[i][i])
            indexs_real = sorted_index_real[torch.linspace(0, 5000-1, samples).to(torch.long)]
            scores_real_sample = scores_real[i][i][indexs_real]
            scores_gen_sample = scores_gen[i][i][indexs_real]
            images_real_sample = [real_images[i][index] for index in indexs_real]
            images_gen_sample = [gen_images[i][index] for index in indexs_real]

            images = all_images[i]
            embed = all_emb[i]
            labels = all_labels[i]
            imdexs = list(range(len(images)))
            images = list(zip(images, labels, imdexs))


            loss_func = ood_detectors[i]


            loss_func.to(device)
            real_mean, real_std = scores_real[i][i].mean(), scores_real[i][i].std()
            threshold = real_mean + 2*real_std
            if verbose:
                print(f"{i} Threshold {threshold:.2f}, Real Mean {real_mean:.2f}, Real Std {real_std:.2f}")
            update_epoch = 2
            images_gen_sample_updated, ood_scores= update_anchors(embed, images, labels, update_epoch, loss_func, generator, encoder, encoder_transform, our_model, model_transform, threshold, batch_size, device)
            scores_gen_sample_updated = ood_scores[-1][indexs_real]
            if verbose:
                fig, ax = plt.subplots(4, samples, figsize=(samples*2, 10))
                for j in range(samples):
                    ax[0, j].imshow(images_real_sample[j].permute(1, 2, 0))
                    ax[0, j].set_title(f"Real {scores_real_sample[j]:.2f}")
                    ax[0, j].axis('off')
                    ax[1, j].imshow(images_gen_sample[j].permute(1, 2, 0))
                    ax[1, j].set_title(f"Gen {scores_gen_sample[j]:.2f}")
                    ax[1, j].axis('off')
                    ax[2, j].imshow(images_gen_sample_updated[indexs_real[j]].permute(1, 2, 0))
                    ax[2, j].set_title(f"Gen Tuned {scores_gen_sample_updated[j]:.2f}")
                    ax[2, j].axis('off')
                    diff = (images_gen_sample_updated[indexs_real[j]] - images_gen_sample[j]).abs()
                    # diff = diff - diff.min()
                    # diff = diff / diff.max()
                    ax[3, j].imshow(diff.permute(1, 2, 0))
                    ax[3, j].set_title(f"Diff")
                    ax[3, j].axis('off')

                plt.savefig(f"figs/{dataset}/{PREFIX}_sample_{ood_detectors[0].name}_{encoder_name}_{generator_name}_{epoch}_{i}.png")
                plt.close()



        # print(f"enbedding diff {(embed - original).abs().sum()}")



def update_anchors(anchors, images, lables, epochs, loss_func, generator, encoder, encoder_transform, our_model, model_transform, batch_size, threshold, device):
    our_model.eval()
    our_model.to(device)

    learnable = [torch.nn.Parameter(a.clone().to(device)) for a in anchors]
    lables = lables.to(device)
    logit_loss = torch.nn.CrossEntropyLoss(reduction='none')
    for p in learnable:
        p.requires_grad = True

    optimizers = [torch.optim.Adam([p], lr=1e-1) for p in learnable]

    out_images = [None]*len(anchors)
    losses = []

    for epoch in range(epochs):
        index_to_update = torch.arange(len(anchors))
        shuffled_index = torch.randperm(len(index_to_update))
        epoch_loss = 0
        losses.append([0]*len(index_to_update))
        for index_batch in shuffled_index.split(batch_size):
            index_batch = index_to_update[index_batch]
            for index in index_batch:
                optimizers[index].zero_grad()
            x = torch.stack([learnable[index] for index in index_batch])
            y = torch.stack([lables[index] for index in index_batch])
            gen_images = generator.generate(x, y)
            model_images = model_transform(gen_images)
            encoder_images = encoder_transform(gen_images)
            # real_images = torch.stack([real_trasform(images[i][0]) for i in index_batch]).to(x.device)
            # image_grid = (torchvision.utils.make_grid(gen_images.detach().cpu(), nrow=8) * std.view(3, 1, 1) + mean.view(3, 1, 1))
            # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            # ax[0].imshow(image_grid.permute(1, 2, 0))
            # image_grid = (torchvision.utils.make_grid(real_images.detach().cpu(), nrow=8) * std.view(3, 1, 1) + mean.view(3, 1, 1))
            # ax[1].imshow(image_grid.permute(1, 2, 0))
            # plt.savefig(f"figs/{PREFIX}_gen_{epoch}.png")
            # exit()

            # batch_image_loss = (real_images - gen_images).abs().mean((1, 2, 3))
            # batch_gen_embs = encoder(gen_images)
            # batch_real_embs = encoder(real_images)
            # batch_emb_loss = (batch_gen_embs - batch_real_embs).abs().mean(1)
            # batch_loss = batch_image_loss + batch_emb_loss

            # batch_loss = batch_image_loss
            logits = our_model(model_images)
            batch_logit_loss = logit_loss(logits, y)
            # batch_loss = -batch_logit_loss

            batch_embs = encoder(encoder_images)
            batch_ood_loss = loss_func(batch_embs)
            all_above_threshold = batch_ood_loss > threshold

            batch_loss = batch_ood_loss / threshold - batch_logit_loss
            # batch_loss[all_above_threshold] = batch_ood_loss[all_above_threshold] / threshold + batch_logit_loss[all_above_threshold]
            # print(batch_loss.shape)
            batch_loss.sum().backward()
            for index, loss, img, ood_loss in zip(index_batch, batch_loss, gen_images, batch_ood_loss):
                optimizers[index].step()
                epoch_loss += loss.item()
                out_images[index] = img.detach().cpu()
                # out_images[index] = resize(img.detach().cpu(), (64, 64))
                losses[-1][index] = ood_loss.item()
        epoch_loss /= len(index_to_update)
        # print(f"Epoch {epoch} Loss: {epoch_loss}")
    for l, a in zip(learnable, anchors):
        a.copy_(l.data)
    our_model.to('cpu')
    return out_images, np.array(losses)




if __name__ == "__main__":
    main()