import ood_detectors.vision as vision_ood
import torch
import pathlib
import project.utils as utils
from collections import defaultdict

class DataSaver:
    def __init__(self, size=1000):
        self.size = size
        self.images = []
        self.embeddings = []
        self.labels = []



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = 'ITGAN'
    dataset = 'CIFAR10'
    model_name = "resnet18.a1_in1k"
    exp = 0
    batch_size = 1
    num_workers = 0

    root_path = pathlib.Path('/mnt/data/arty/data/IT-GAN')
    data_path = root_path / 'data' 
    checkpoint_path = root_path / 'checkpoints' 
    weight_path = checkpoint_path / f"G_Pretrained_{dataset}_exp{exp}.pth"

    encoder = vision_ood.get_encoder("dinov2")
    encoder.eval()
    encoder.to(device)
    encoder_name = encoder.name

    data = utils.get_dataset(dataset, data_path, None)
    anchors = utils.load_anchors(checkpoint_path / f'Base_ITGAN_{dataset}_exp{exp}.pt')
    generator, mean, std = utils.load_generator(*anchors[2:], weight_path)
    train_img = data["train"]
    anchor_emb = anchors[0]
    label = anchors[1]
    generator_name = anchors[2]
    config = anchors[3]
    data_save = DataSaver()

    data_blob = defaultdict(list)
    for img, l, i in train_img:
        emb = anchor_emb[i]
        data_blob[int(l)].append(((img, l, i), emb))

    for k, v in data_blob.items():
        images, embeddings = zip(*v)
        labes = torch.tensor([k] * len(images))
        embeddings = torch.stack(embeddings)

        gen_loader = utils.TransformLoader(utils.GeneratorDatasetLoader(embeddings, labes, generator_name, config, weight_path, shuffle=False, num_workers=0, batch_size=batch_size, device=device, use_cache=False), utils.get_transforms(encoder, 'Encoder'), device=device)
        real_loader = utils.TransformLoader(torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.imge_stack), utils.get_transforms(encoder, 'Real'), device=device)

        for (gen_emb, l, index), (real_img, *_) in zip(gen_loader, real_loader):
            print(gen_emb.shape, real_img.shape)
            break
            

    

if __name__ == "__main__":
    main()