"""T-SNE visualization of latent space embeddings.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import json
import torch

import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
import utils.logger as utils_logger
import utils.data as utils_data
import utils.paths as utils_paths

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from options.training import parser
from dataset.get_datasets import get_datasets
from models.get_model import get_model
from matplotlib import font_manager

font_manager.fontManager.addfont('/mnt/data/hyx/projects/CktGen-Exp/assets/arial.ttf')

def plot_one_tsne(args, ckt_latents, labels, output_path):
    """Plots t-SNE visualization of circuit latent embeddings.
    
    Reduces high-dimensional circuit latents to 2D using t-SNE and creates
    a scatter plot colored by cluster labels. Used for visualizing latent
    space structure of unconditional models (VAE, CVAEGAN without spec latents).
    
    Args:
        args: Configuration dict with 'modeltype' and 'archiname' for filename.
        ckt_latents: Circuit latent embeddings, shape (N, latent_dim).
        labels: Cluster labels for each latent, shape (N,). Range: 0-9.
        output_path: Path to save the output SVG file.
        
    Notes:
        - Uses tab10 colormap for up to 10 distinct clusters
        - Saves plot as SVG to output_path
        - Hides axes for cleaner visualization
    """
    plt.rcParams['font.family'] = 'arial'

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(ckt_latents)

    # Visualization
    cmap = plt.get_cmap('tab10')  # 10 main colors

    plt.figure(figsize=(9,6))
    for i in range(10):
        # Data latent (light color/marker1)
        idx = (labels == i)
        plt.scatter(latents_2d[idx,0], latents_2d[idx,1], color=cmap(i, 0.5), alpha=0.5, s=120, marker='o')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path



def plot_two_tsne(args, ckt_latents, spec_latents, labels, spec_labels, output_path):
    """Plots t-SNE visualization of both circuit and specification latent embeddings.
    
    Reduces high-dimensional circuit and specification latents to 2D using t-SNE 
    and creates a scatter plot showing alignment between circuit data (circles) 
    and specification embeddings (plus signs) colored by cluster. Used for 
    conditional models like CktGen and Evaluator.
    
    Args:
        args: Configuration dict with 'modeltype' and 'archiname' for filename.
        ckt_latents: Circuit latent embeddings, shape (1000, latent_dim).
        spec_latents: Specification latent embeddings, shape (1000, latent_dim).
        labels: Cluster labels for latents, shape (2000,). Range: 0-9.
        spec_labels: Currently unused (for future extension).
        output_path: Path to save the output SVG file.
        
    Notes:
        - Combines 1000 circuit + 1000 spec latents into single (2000, dim) array
        - Circuit latents shown as circles (alpha=0.5, size=120)
        - Spec latents shown as plus signs (alpha=0.8, size=150)
        - Uses tab10 colormap for up to 10 distinct clusters
        - Saves plot as SVG to output_path
    """
    plt.rcParams['font.family'] = 'arial'

    all_latents = np.concatenate([ckt_latents, spec_latents], axis=0)  # (2000, latent_dim)

    types = np.array([0]*1000 + [1]*1000)  # 0=data, 1=class latent
    types = np.array(types)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(all_latents)

    # Visualization
    cmap = plt.get_cmap('tab10')  # 10 main colors

    plt.figure(figsize=(9,6))
    for i in range(10):
        # Data latent (light color/marker1)
        idx = np.where((labels == i) & (types == 0))[0]
        plt.scatter(latents_2d[idx,0], latents_2d[idx,1], color=cmap(i, 0.5), label=f'Class {i} data', alpha=0.5, s=120, marker='o')

        # Class latent (dark color/marker2)
        idx_class = np.where((labels == i) & (types == 1))[0]
        plt.scatter(latents_2d[idx_class,0], latents_2d[idx_class,1], color=cmap(i, 1.0), label=f'Class {i} class_latent', alpha=0.8, s=150, marker='+', linewidths=0.7)

    # Only show legend once (deduplicate)
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels_leg, handles))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path



def tsne_on_datasets(args, model, datasets, output_path):
    """Generates t-SNE visualizations of learned latent spaces from datasets.
    
    Samples circuits from the top 10 specification clusters, encodes them using
    the model, and generates t-SNE plots. For conditional models (CktGen, Evaluator),
    plots both circuit and specification latents. For unconditional models (VAE, 
    CVAEGAN, LDT), plots only circuit latents.
    
    Args:
        args: Configuration dictionary with 'modeltype' for determining visualization type.
        model: Trained model with encoding capabilities (must be in eval mode).
        datasets: Dictionary with 'train' and 'test' dataset splits.
        output_path: Path to save the output SVG file.
        
    Returns:
        str: Path to the saved output file.
        
    Notes:
        - Clusters circuits by specification domains
        - Samples 100 circuits from each of the top 10 clusters (1000 total)
        - CktGen: Encodes to both circuit and spec latents
        - Evaluator: Generates circuit and spec embeddings
        - VAE/CVAEGAN/LDT: Encodes to circuit latents only
    """
    all_specs = utils_data.get_specifications(datasets['test'])

    dataset_all = datasets['train'] + datasets['test']
    specs_clustered_ckts = utils_data.get_specification_domain(dataset_all)
    sorted_cluster = sorted(specs_clustered_ckts, key=lambda x: len(x), reverse=True)
    spec_labels = []
    all_ckt_latents = []
    all_spec_latents = []
    cluster_num = 10
    sample_numbers = 100

    for i in range(0, cluster_num):
        ckts = sorted_cluster[i][:sample_numbers]
        batch = utils_data.transforms(args, ckts)

        if args['modeltype']=='cktgen':
            spec_latents = model.sample_spec_latents(batch, sample_mean=False, return_dists=False)
            ckt_latents = model.sample_ckt_latents(batch, sample_mean=False, return_dists=False)
            all_ckt_latents.append(ckt_latents.cpu().detach().numpy())
            all_spec_latents.append(spec_latents.cpu().detach().numpy())
        elif args['modeltype'] == 'evaluator':
            ckt_embs = model.get_ckt_embeddings(batch)
            spec_embs = model.get_spec_embeddings(batch)
            all_ckt_latents.append(ckt_embs.cpu().detach().numpy())
            all_spec_latents.append(spec_embs.cpu().detach().numpy())
        else:
            if args['modeltype'] == 'ldt':
                ckt_latents = model(args, batch).squeeze(1)
            else:
                ckt_latents = model.sample_ckt_latents(batch, sample_mean=False, return_dists=False)
            all_ckt_latents.append(ckt_latents.cpu().detach().numpy())
        
    
    if args['modeltype']=='cktgen' or args['modeltype']=='evaluator':
        labels = [i // 100 for i in range(1000)]
        labels = labels + labels
        flat_ckt_latents = np.concatenate(all_ckt_latents, axis=0)
        flat_spec_latents = np.concatenate(all_spec_latents, axis=0)
        return plot_two_tsne(args, flat_ckt_latents, flat_spec_latents, np.array(labels), spec_labels, output_path)
    else:
        labels = [i // 100 for i in range(1000)]
        flat_ckt_latents = np.concatenate(all_ckt_latents, axis=0)
        return plot_one_tsne(args, flat_ckt_latents, np.array(labels), output_path)


def main():
    """Main entry point for generating t-SNE visualizations of model latent spaces.
    
    Parses configuration, loads datasets and trained model from checkpoint,
    then generates t-SNE visualizations showing the structure of learned
    latent embeddings.
    
    The function:
        - Sets up output directory and logger
        - Loads datasets (train + test)
        - Initializes model architecture
        - Loads trained model from checkpoint (required via 'resume_pth')
        - Generates t-SNE plots of latent space structure
        
    Requires:
        - 'resume_pth' must be specified in args to load pretrained model
        - Model must be trained before visualization
        
    Outputs:
        - SVG plot showing t-SNE projection of latent embeddings
        - Output path: {out_dir}/{exp_name}.svg
    """
    args = parser()

    ####--- Setup Paths (Centralized Path Management) ---####
    utils_paths.setup_paths(args)
    os.makedirs(args['out_dir'], exist_ok=True)

    ####--- Logger ---####
    logger = utils_logger.get_logger(args['out_dir'], args.get('exp_name'))
    logger.info(json.dumps(args, indent=4, sort_keys=True, default=utils_logger.serialize))
    
    ####--- Dataset ---####
    datasets = get_datasets(args)
    model = get_model(args)

    # Build output path from out_dir and exp_name
    exp_name = args.get('exp_name', 'tsne')
    out_dir = Path(args['out_dir'])
    output_path = out_dir / f"{exp_name}.svg"

    if 'resume_pth' in args:
        logger.info('Loading model from {}'.format(args['resume_pth']))
        model = torch.load(args['resume_pth'], map_location='cpu').to(args['device'])
        model.eval()

        saved_path = tsne_on_datasets(args, model, datasets, str(output_path))
        
        # Print completion message
        print("\n" + "=" * 60)
        print("t-SNE visualization completed successfully!")
        print(f"Output file: {saved_path}")
        print("=" * 60 + "\n")
    else:
        logger.info('###############################################################################')
        logger.info('              Need to load the pretrained model to do the encoding')
        logger.info('###############################################################################')


if __name__ == "__main__":
    main()