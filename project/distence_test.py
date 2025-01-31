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
import seaborn as sns



def plot(id_score, gen_score, oods, dataset, tile, out_dir='figs', verbose=True, names=None):
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

    ax.set_title('Density Plots')
    ax.set_xlabel('bits/dim')
    ax.set_ylabel('Density')

    ax.legend()


    # Save the figure
    out_dir = pathlib.Path(out_dir) / dataset
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{tile}.svg"
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
    batch_size = 64
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
    generator, mean, std = utils.load_generator(*anchors[2:], weight_path)
    train_img = data["train"]
    anchor_emb = anchors[0]
    label = anchors[1]
    generator_name = anchors[2]
    config = anchors[3]

    data_blob = defaultdict(list)

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

            real_loader = utils.TransformLoader(torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack), utils.get_transforms(encoder, 'Real'), device=device)

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

            gen_loader = utils.TransformLoader(utils.GeneratorDatasetLoader(embeddings, labels, generator_name, config, weight_path, shuffle=False, num_workers=num_workers, batch_size=batch_size, device=device, use_cache=False), utils.get_transforms(encoder, 'Encoder'), device=device)

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

    ood_detectors = [likelihood.RDM(embedding_size).to("cpu") for _ in range(10)]
    print("Fitting OOD Detectors")
    for i, ood_detector in tqdm.tqdm(list(enumerate(ood_detectors))):
        if pathlib.Path(checkpoint_path/f"ood_detector_{dataset}_{encoder_name}_{generator_name}_{i}.pt").exists() and not clean:
            ood_detector.load_state_dict(torch.load(checkpoint_path/f"ood_detector_{dataset}_{encoder_name}_{generator_name}_{i}.pt", weights_only=False))
            print(f"OOD Detector {i} Loaded")
        else:
            ood_detector.to(device)
            ood_detector.fit(real_emb[i], batch_size=1000, n_epochs=1000, verbose=False)
            ood_detector.to("cpu")
            torch.save(ood_detector.state_dict(), checkpoint_path/f"ood_detector_{dataset}_{encoder_name}_{generator_name}_{i}.pt")
            print(f"OOD Detector {i} Fitted")
    scores_real = torch.zeros((10, 10, 5000))
    scores_gen = torch.zeros((10, 10, 5000))

    print("Calculating Scores")
    for i, ood_detector in tqdm.tqdm(list(enumerate(ood_detectors))):
        for index in range(10):
            if pathlib.Path(checkpoint_path / f'score_real_{i}_{index}_{dataset}_{encoder_name}.pt').exists() and not clean:
                scores_real[i][index] = torch.load(checkpoint_path / f'score_real_{i}_{index}_{dataset}_{encoder_name}.pt', weights_only=False)
            else:
                ood_detector.to(device)
                score_real = ood_detector.predict(real_emb[index], batch_size=2048, verbose=False)
                ood_detector.to("cpu")
                score_real = torch.from_numpy(score_real)
                torch.save(score_real, checkpoint_path / f'score_real_{i}_{index}_{dataset}_{encoder_name}.pt')
                scores_real[i][index] = score_real
            if pathlib.Path(checkpoint_path / f'score_gen_{i}_{index}_{dataset}_{encoder_name}.pt').exists() and not clean:
                scores_gen[i][index] = torch.load(checkpoint_path / f'score_gen_{i}_{index}_{dataset}_{encoder_name}.pt', weights_only=False)
            else:
                ood_detector.to(device)
                score_gen = ood_detector.predict(gen_emb[index], batch_size=2048, verbose=False)
                score_gen = torch.from_numpy(score_gen)
                ood_detector.to("cpu")
                torch.save(score_gen, checkpoint_path / f'score_gen_{i}_{index}_{dataset}_{encoder_name}.pt')
                scores_gen[i][index] = score_gen

    print("Plotting")
    for i in range(10):
        ood_scores = [scores_real[i][j].numpy() for j in range(10) if j != i] + [scores_gen[i][j].numpy() for j in range(10) if j != i]
        names = [f"{j} Real" for j in range(10) if j != i] + [f"{j} Gen" for j in range(10) if j != i]
        plot(scores_real[i][i].numpy(), scores_gen[i][i].numpy(), ood_scores, dataset, f"{encoder_name}_{generator_name}_{i}", names=names)
        samples = 10
        sorted_index_real = torch.argsort(scores_real[i][i])
        sorted_index_gen = torch.argsort(scores_gen[i][i])
        indexs_real = sorted_index_real[torch.linspace(0, 5000-1, samples).to(torch.long)]
        indexs_gen = sorted_index_gen[torch.linspace(0, 5000-1, samples).to(torch.long)]
        scores_real_sample = scores_real[i][i][indexs_real]
        scores_gen_sample = scores_gen[i][i][indexs_gen]
        images_real_sample = [real_images[i][index] for index in indexs_real]
        images_gen_sample = [gen_images[i][index] for index in indexs_gen]

        fig, ax = plt.subplots(2, samples, figsize=(samples*2, 5))
        for j in range(samples):
            ax[0, j].imshow(images_real_sample[j].permute(1, 2, 0))
            ax[0, j].set_title(f"Real {scores_real_sample[j]:.2f}")
            ax[0, j].axis('off')
            ax[1, j].imshow(images_gen_sample[j].permute(1, 2, 0))
            ax[1, j].set_title(f"Gen {scores_gen_sample[j]:.2f}")
            ax[1, j].axis('off')
        plt.savefig(f"figs/{dataset}/sample_{encoder_name}_{generator_name}_{i}.svg")


        


        




            

    

if __name__ == "__main__":
    main()