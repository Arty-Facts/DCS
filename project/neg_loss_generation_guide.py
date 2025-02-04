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
from torch.autograd import gradcheck
import torchvision
from baseline_model import get_baseline_model
import yaml

PREFIX = "nlgg"

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = 'ITGAN'
    dataset = 'CIFAR10'
    model_name = "resnet18.a1_in1k"
    exp = 0
    batch_size = 8
    num_workers = 0
    clean = False

    root_path = pathlib.Path('/mnt/data/arty/data/IT-GAN')
    data_path = root_path / 'data' 
    model_path = root_path / 'checkpoints'
    checkpoint_path = pathlib.Path('checkpoints')
    checkpoint_path.mkdir(exist_ok=True, parents=True)
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
    
    base_model = get_baseline_model(baseline_model_config)
    base_model.to(device)
    base_model.eval()
    

    real_emb = torch.zeros((10, 5000, embedding_size))
    gen_emb = torch.zeros((10, 5000, embedding_size))

    real_images = [[None]*5000 for _ in range(10)]
    gen_images = [[None]*5000 for _ in range(10)]
    for img, l, i in train_img:
        emb = anchor_emb[i]
        data_blob[int(l)].append((img, l, emb))
    if pathlib.Path(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and pathlib.Path(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and not clean:
        print("Loading real embeddings")
        real_emb = torch.load(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
        real_images = torch.load(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
    else:
        print("Calculating real embeddings")
        for l, v in tqdm.tqdm(data_blob.items()):
            images, labels, embeddings = zip(*v)
            indexs = list(range(len(images)))
            embeddings = torch.stack(embeddings)
            images = list(zip(images, labels, indexs))
            labels = torch.tensor(labels)

            real_loader = utils.ImageLoader(torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack), utils.get_transforms(encoder, 'Real'), device=device)

            for (real_img, li, index) in real_loader:
                for idx, ri in zip(index, real_img):
                    real_images[int(l)][idx] = resize(unnormalize(ri).detach().cpu(), (64, 64))
                real_emb[li, index] = encoder(real_img).detach().cpu()
        with open(checkpoint_path/f"real_emb_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(real_emb, f)
        with open(checkpoint_path/f"real_images_{dataset}_{encoder_name}_{generator_name}.pt", 'wb') as f:
            torch.save(real_images, f)
    
    if pathlib.Path(checkpoint_path/f"gen_emb_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and pathlib.Path(checkpoint_path/f"gen_images_{dataset}_{encoder_name}_{generator_name}.pt").exists() \
        and not clean:
        print("Loading generated embeddings")
        gen_emb = torch.load(checkpoint_path/f"gen_emb_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
        gen_images = torch.load(checkpoint_path/f"gen_images_{dataset}_{encoder_name}_{generator_name}.pt", weights_only=False)
    else:
        print("Calculating generated embeddings")
        for l, v in tqdm.tqdm(data_blob.items()):
            images, labels, embeddings = zip(*v)
            indexs = list(range(len(images)))
            embeddings = torch.stack(embeddings)
            images = list(zip(images, labels, indexs))
            labels = torch.tensor(labels)

            gen_loader = utils.TransformLoader(utils.GeneratorDatasetLoader(embeddings, labels, generator_name, config, weight_path, shuffle=False, num_workers=num_workers, batch_size=batch_size, device=device, use_cache=False), utils.get_transforms(encoder, 'Encoder', device=device), device=device)

            for (gen_img, li, index) in gen_loader:
                for idx, gi in zip(index, gen_img):
                    gen_images[int(l)][idx] = resize(unnormalize(gi).detach().cpu(), (64, 64))
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
    plt.savefig(f"pca_{dataset}_{encoder_name}_{generator_name}.svg")

    # ood_detectors = [likelihood.RDM(embedding_size).to("cpu") for _ in range(10)]
    ood_detectors = [residual.Residual(0.3) for _ in range(10)]
    print("Fitting OOD Detectors")
    for i, ood_detector in tqdm.tqdm(list(enumerate(ood_detectors))):
        if pathlib.Path(checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt").exists() and not clean:
            ood_detector.load_state_dict(torch.load(checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt", weights_only=False))
            print(f"OOD Detector {i} Loaded")
        else:
            ood_detector.to(device)
            ood_detector.fit(real_emb[i], batch_size=1000, n_epochs=1000, verbose=False)
            ood_detector.to("cpu")
            torch.save(ood_detector.state_dict(), checkpoint_path/f"{ood_detector.name}_{dataset}_{encoder_name}_{generator_name}_{i}.pt")
            print(f"OOD Detector {i} Fitted")
    scores_real = torch.zeros((10, 10, 5000))
    scores_gen = torch.zeros((10, 10, 5000))

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

    print("Plotting")
    for i in range(10):
        ood_scores = [scores_real[i][j].numpy() for j in range(10) if j != i] + [scores_gen[i][j].numpy() for j in range(10) if j != i]
        names = [f"{j} Real" for j in range(10) if j != i] + [f"{j} Gen" for j in range(10) if j != i]

        plot(scores_real[i][i].numpy(), scores_gen[i][i].numpy(), ood_scores, dataset, f"{PREFIX}_disp_plot_{ood_detectors[0].name}_{encoder_name}_{generator_name}_{i}", names=names)
        samples = 10
        sorted_index_real = torch.argsort(scores_real[i][i])
        sorted_index_gen = torch.argsort(scores_gen[i][i])
        indexs_real = sorted_index_real[torch.linspace(0, 5000-1, samples).to(torch.long)]
        indexs_gen = sorted_index_gen[torch.linspace(0, 5000-1, samples).to(torch.long)]
        scores_real_sample = scores_real[i][i][indexs_real]
        scores_gen_sample = scores_gen[i][i][indexs_real]
        images_real_sample = [real_images[i][index] for index in indexs_real]
        images_gen_sample = [gen_images[i][index] for index in indexs_real]

        curr_data = data_blob[i]
        images, labels, embeddings = zip(*curr_data)
        embed = torch.stack(embeddings).clone()
        original = embed.clone()
        imdexs = list(range(len(images)))
        images = list(zip(images, labels, imdexs))
        labels = torch.tensor(labels)
        

        loss_func = ood_detectors[i]


        loss_func.to(device)
        real_mean, real_std = scores_real[i].mean(), scores_real[i].std()
        threshold = real_mean + 2*real_std
        generator = utils.GeneratorDatasetLoader(*utils.load_anchors(model_path / f'Base_ITGAN_{dataset}_exp{exp}.pt'), weight_path, shuffle=False, num_workers=num_workers, batch_size=batch_size,  device=device, use_cache=False)
        update_epoch = 5
        images_gen_sample_updated, ood_scores= update_anchors(embed, images, labels, update_epoch, loss_func, generator, encoder, base_model, threshold, batch_size, device)
        names = [f"epoch_{i}" for i in range(1, update_epoch)]
        plot(scores_real[i][i].numpy(), scores_gen[i][i].numpy(), ood_scores[1:],  dataset, f"{PREFIX}_tuned_disp_plot_{ood_detectors[0].name}_{encoder_name}_{generator_name}_{i}_gen", names=names)
        scores_gen_sample_updated = ood_scores[-1][indexs_real]

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

        plt.savefig(f"figs/{dataset}/{PREFIX}_sample_{ood_detectors[0].name}_{encoder_name}_{generator_name}_{i}.svg")
        
        # print(f"enbedding diff {(embed - original).abs().sum()}")
        


def update_anchors(anchors, images, lables, epochs, loss_func, generator, encoder, base_model, batch_size, threshold, device):
    learnable = [torch.nn.Parameter(a.clone().to(device)) for a in anchors]
    lables = lables.to(device)
    logit_loss = torch.nn.CrossEntropyLoss(reduction='none')
    for p in learnable:
        p.requires_grad = True

    optimizers = [torch.optim.Adam([p], lr=1e-1) for p in learnable]
    real_trasform = utils.get_transforms(encoder, 'Real')
    gen_transform = utils.get_transforms(encoder, 'Encoder').to(device)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # real_trasform = lambda x: (torchvision.transforms.functional.pil_to_tensor(x).to(dtype=torch.float32, device=device)) / 255
    # gen_transform = lambda x: x

    out_images = [None]*len(anchors)
    losses = []

    for epoch in tqdm.tqdm(range(epochs)):
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
            gen_images = gen_transform(gen_images)
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
            logits = base_model(gen_images)
            batch_logit_loss = logit_loss(logits, y)
            batch_loss = -batch_logit_loss
            with torch.no_grad():
                batch_embs = encoder(gen_images)
                batch_ood_loss = loss_func(batch_embs) 
            # batch_loss = batch_ood_loss / threshold - batch_logit_loss
            # print(batch_loss.shape)
            batch_loss.sum().backward()
            for index, loss, img, ood_loss in zip(index_batch, batch_loss, gen_images, batch_ood_loss):
                optimizers[index].step()
                epoch_loss += loss.item()
                out_images[index] = resize(unnormalize(img.detach()).cpu(), (64, 64))
                # out_images[index] = resize(img.detach().cpu(), (64, 64))
                losses[-1][index] = ood_loss.item()
        epoch_loss /= len(index_to_update)
        print(f"Epoch {epoch} Loss: {epoch_loss}")
    for l, a in zip(learnable, anchors):
        a.copy_(l.data)
    return out_images, np.array(losses)

       


if __name__ == "__main__":
    main()