import os
import sys
sys.path.append('/home/ljc/works/diffusion-dg-reid')
import os.path as osp
import tqdm
import argparse
import torch
import pickle
from diffusers import AutoencoderKL
from engine.datasets.dataloader import make_dataloader_vae_extract

def main(args):
    device = 'cuda'
    loader = make_dataloader_vae_extract(args.dataset_name, args.data_path, args.image_size,
                                         args.batchsize, args.num_workers)
    vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-mse').to(device)
    
    feature_dict = {}
    features = []
    fnames = []
    for i, (path, img, *_) in enumerate(tqdm.tqdm(loader, desc='Extract VAE features')):
        img = img.to(device)
        with torch.no_grad():
            x = vae.encode(img).latent_dist.sample().mul_(0.18215)
        x = x.detach().cpu()
        features.append(x)
        fnames.extend(path)
    features = torch.cat(features, dim=0).numpy()
    
    for x, y in zip(features, fnames):
        feature_dict[y] = x
    
    os.makedirs(args.results_dir, exist_ok=True)
    with open(f'{args.results_dir}/{args.dataset_name}_sd-vae-ft-mse_{args.image_size}x{args.image_size}_feature_dict.pkl', 'wb') as f:
        pickle.dump(feature_dict, f)

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)