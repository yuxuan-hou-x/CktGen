<div align="center">

# CktGen: Automated Analog Circuit Design with Generative Artificial Intelligence

[![Paper](https://img.shields.io/badge/Paper-Engineering-blue)](https://www.sciencedirect.com/science/article/pii/S2095809925008148)
[![arXiv](https://img.shields.io/badge/arXiv-2410.00995-b31b1b.svg)](https://arxiv.org/abs/2410.00995)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Yuxuan-Hou/CktGen)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/teaser.png" width="600">

</div>

**CktGen** is a specification-conditioned generative framework for automated analog circuit design. Given target performance specifications (gain, bandwidth, phase margin), CktGen generates diverse valid circuits through:

**(a) Joint Representation Learning**: Maps circuits and specifications into a canonical latent space using contrastive learning and classifier guidance, enabling one-to-many mappings from specifications to circuits.

**(b) Test-Time Optimization**: Searches for optimal circuits meeting target specifications using a multi-armed bandit algorithm — **no retraining required** when design requirements change.

If you find this work useful, please cite our paper:

```bibtex
@article{hou2025cktgen,
  title = {CktGen: Automated Analog Circuit Design with Generative Artificial Intelligence},
  journal = {Engineering},
  year = {2025},
  issn = {2095-8099},
  doi = {https://doi.org/10.1016/j.eng.2025.12.025},
  url = {https://www.sciencedirect.com/science/article/pii/S2095809925008148},
  author = {Yuxuan Hou and Hehe Fan and Jianrong Zhang and Yue Zhang and Hua Chen and Min Zhou and Faxin Yu and Roger Zimmermann and Yi Yang},
}
```

## 📋 Table of Contents

- [Project Structure](#️-project-structure)
- [Installation](#️-installation)
- [Datasets](#-datasets)
- [Evaluate with Pre-Trained models](#evaluate-with-pre-trained-models)
- [Train from scratch](#️-train-from-scratch)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🏗️ Project Structure

<details>
<summary><b>📁 Click to expand</b></summary>

```
CktGen/
├── checkpoints/              # Pre-trained models (download from HuggingFace)
│   ├── cktgen/               # CktGen models
│   ├── evaluator/            # Performance evaluators
│   └── baselines/            # Baseline models (CktGNN, LDT, PACE, CVAEGAN)
├── dataset/OCB/              # Open Circuit Benchmark datasets
│   ├── CktBench101/          # 10k circuit samples
│   └── CktBench301/          # 50k circuit samples
├── models/                   # Model implementations
│   ├── architectures/        # Encoders & decoders
│   └── modeltype/            # VAE, CktGen, LDT, CVAEGAN
├── scripts/                  # Experiment scripts
│   ├── train/                # Training scripts
│   └── test/                 # Evaluation scripts
├── train/                    # Training source code
├── test/                     # Testing source code
├── evaluation/               # Evaluation metrics
├── utils/                    # Utility functions (including model download)
└── options/                  # CLI argument definitions
```

</details>

## 🛠️ Installation

### Prerequisites

- **Hardware**: NVIDIA GPU with 24GB+ VRAM recommended (tested on RTX 4090)
- **Software**: [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Install environment

The `environment.yml` file contains all required dependencies (Python 3.9, PyTorch 1.13.1, CUDA 11.7, etc.) and has been tested to work out-of-the-box.

```bash
# Step 1: Clone the repository
git clone https://github.com/hhyxx/CktGen.git
cd CktGen

# Step 2: Create and activate environment (all dependencies included)
conda env create -f environment.yml
conda activate cktgen
```

## 📊 Datasets

All experiments in this work are conducted on the **Open Circuit Benchmark (OCB)** dataset. To facilitate easy reproducibility and provide a seamless "ready-to-run" experience, we have included the complete OCB datasets directly within this repository.

### 🔗 Dataset Credits & Citation

- The OCB dataset was originally introduced as a foundational benchmark by the authors of **CktGNN** ([Dong et al., 2023](https://github.com/zehao-dong/CktGNN)). We are deeply grateful for their pioneering contribution to the analog circuit design community, which established essential standards for benchmarking in this field.

- When using this dataset, please cite the original CktGNN paper to properly acknowledge the foundational work of its authors. The data is redistributed here under the [MIT License](https://github.com/zehao-dong/CktGNN/blob/main/LICENSE) for research convenience.

### 📂 Dataset Structure

```text
dataset/OCB/
├── CktBench101/                   # Ckt-Bench-101 analog circuits
│   ├── ckt_bench_101_igraph.pkl   # Circuit graphs (10k samples)
│   └── perform101.csv             # Performance specifications
└── CktBench301/                   # Ckt-Bench-301 analog circuits
    ├── ckt_bench_301_igraph.pkl   # Circuit graphs (50k samples)
    └── perform301.csv             # Performance specifications
```

## 📈 Evaluate with Pre-Trained models

### Download Pre-trained Models

To run the evaluation tests, you need to download the pre-trained models:

1. **Download from [Hugging Face](https://huggingface.co/Yuxuan-Hou/CktGen)** (No authentication required):

   💡 **Tip**: Run `python utils/load_pretrained.py --list` to see all available files.

   <table>
   <tr><th>Model</th><th>Command</th></tr>
   <tr><td><b>All checkpoints</b></td><td><code>python utils/load_pretrained.py</code></td></tr>
   <tr><td>CktGen</td><td><code>python utils/load_pretrained.py --folder cktgen</code></td></tr>
   <tr><td>Evaluator</td><td><code>python utils/load_pretrained.py --folder evaluator</code></td></tr>
   </table>

   <details>
   <summary>📦 <b>Download individual baseline models:</b></summary>
   <br>
   <table>
   <tr><th>Baseline Model</th><th>Command</th></tr>
   <tr><td>Baselines (all)</td><td><code>python utils/load_pretrained.py --folder baselines</code></td></tr>
   <tr><td>LDT (Latent Diffusion Transformer)</td><td><code>python utils/load_pretrained.py --folder baselines/ldt</code></td></tr>
   <tr><td>CVAEGAN (Conditional VAE-GAN)</td><td><code>python utils/load_pretrained.py --folder baselines/cvaegan</code></td></tr>
   <tr><td>CktGNN</td><td><code>python utils/load_pretrained.py --folder baselines/cktgnn</code></td></tr>
   <tr><td>PACE</td><td><code>python utils/load_pretrained.py --folder baselines/pace</code></td></tr>
   </table>

   </details>
   <br>

2. **Download from [Baidu Netdisk](https://pan.baidu.com/s/1mFmhRHf7_qIT5AOiTWoq-g)**:

   Download and extract the folders into `checkpoints/` to use with the provided scripts:

   ```
   checkpoints/
   ├── cktgen/
   ├── evaluator/
   └── baselines/
       ├── ldt/
       └── ...
   ```

### Evaluating CktGen

After downloading the pretrained models into `checkpoints` folder, run these test scripts to reproduce the experimental results of CktGen in our paper:

> ⚠️ **Note**: Our experiments were conducted on **NVIDIA RTX 4090**. Due to differences in GPU architecture and floating-point precision, results on other hardware may vary slightly.

<table>
<tr>
  <th>Experiment</th>
  <th>Dataset</th>
  <th>Command</th>
</tr>
<tr>
  <td rowspan="2"><b>Auto Design</b></td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/cktgen/cktgen_auto_design_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/cktgen/cktgen_auto_design_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2"><b>Conditional Generation</b></td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/cktgen/cktgen_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/cktgen/cktgen_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2"><b>Reconstruction<br>& Random Generation</b></td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/cktgen/cktgen_recon_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/cktgen/cktgen_recon_301.sh</code></td>
</tr>
<tr>
  <td><b>t-SNE Visualization</b></td>
  <td>-</td>
  <td><code>bash scripts/test/cktgen/cktgen_tsne.sh</code></td>
</tr>
<tr>
  <td><b>Auto Design Visualization</b></td>
  <td>-</td>
  <td><code>bash scripts/test/auto_design_visualize.sh</code></td>
</tr>
</table>

### Evaluating Baselines

<details>
<summary>📦 <b>Baseline models evaluation scripts</b></summary>
<br>

<table>
<tr>
  <th>Model</th>
  <th>Experiment</th>
  <th>Dataset</th>
  <th>Command</th>
</tr>
<tr>
  <td rowspan="6"><b>CktGNN</b></td>
  <td rowspan="2">Auto Design</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_auto_design_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_auto_design_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td>Reconstruction</td>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_recon_301.sh</code></td>
</tr>
<tr>
  <td>t-SNE</td>
  <td>-</td>
  <td><code>bash scripts/test/baselines/cktgnn/cktgnn_tsne.sh</code></td>
</tr>
<tr>
  <td rowspan="5"><b>LDT</b></td>
  <td rowspan="2">Auto Design</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/ldt/ldt_auto_design_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/ldt/ldt_auto_design_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/ldt/ldt_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/ldt/ldt_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td>t-SNE</td>
  <td>-</td>
  <td><code>bash scripts/test/baselines/ldt/ldt_tsne.sh</code></td>
</tr>
<tr>
  <td rowspan="7"><b>PACE</b></td>
  <td rowspan="2">Auto Design</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/pace/pace_auto_design_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/pace/pace_auto_design_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/pace/pace_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/pace/pace_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Reconstruction</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/pace/pace_recon_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/pace/pace_recon_301.sh</code></td>
</tr>
<tr>
  <td>t-SNE</td>
  <td>-</td>
  <td><code>bash scripts/test/baselines/pace/pace_tsne.sh</code></td>
</tr>
<tr>
  <td rowspan="5"><b>CVAEGAN</b></td>
  <td rowspan="2">Auto Design</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/cvaegan/cvaegan_auto_design_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/cvaegan/cvaegan_auto_design_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/test/baselines/cvaegan/cvaegan_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/test/baselines/cvaegan/cvaegan_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td>t-SNE</td>
  <td>-</td>
  <td><code>bash scripts/test/baselines/cvaegan/cvaegan_tsne.sh</code></td>
</tr>
</table>

</details>

## 🏋️ Train from scratch

### Training CktGen

**Prerequisites**: Training CktGen with conditional generation requires a pre-trained evaluator model for guided optimization. The evaluator must be downloaded and placed in `checkpoints/` before training:

```bash
python utils/load_pretrained.py --folder evaluator
```

Alternatively, train your own evaluator using scripts in `scripts/train/evaluator/`.
With the evaluator loaded, the training script will automatically evaluate performance at the end of training.

<table>
<tr>
  <th>Training Task</th>
  <th>Dataset</th>
  <th>Command</th>
</tr>
<tr>
  <td rowspan="2"><b>Conditional Generation</b><br><i>(requires evaluator)</i></td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/cktgen/cktgen_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/cktgen/cktgen_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2"><b>Reconstruction Only</b><br><i>(no evaluator needed)</i></td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/cktgen/cktgen_recon_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/cktgen/cktgen_recon_301.sh</code></td>
</tr>
</table>

### Training Evaluator

<table>
<tr>
  <th>Dataset</th>
  <th>Command</th>
</tr>
<tr>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/evaluator/evaluator_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/evaluator/evaluator_301.sh</code></td>
</tr>
</table>

### Training Baselines

<details>
<summary>📦 <b>Baseline models training scripts</b></summary>
<br>

<table>
<tr>
  <th>Model</th>
  <th>Training Task</th>
  <th>Dataset</th>
  <th>Command</th>
</tr>
<tr>
  <td rowspan="3"><b>CktGNN</b></td>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/baselines/cktgnn/cktgnn_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/cktgnn/cktgnn_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td>Reconstruction</td>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/cktgnn/cktgnn_recon_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2"><b>LDT</b></td>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/baselines/ldt/ldt_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/ldt/ldt_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td rowspan="4"><b>PACE</b></td>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/baselines/pace/pace_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/pace/pace_cond_gen_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2">Reconstruction</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/baselines/pace/pace_recon_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/pace/pace_recon_301.sh</code></td>
</tr>
<tr>
  <td rowspan="2"><b>CVAEGAN</b></td>
  <td rowspan="2">Conditional Generation</td>
  <td>CktBench-101</td>
  <td><code>bash scripts/train/baselines/cvaegan/cvaegan_cond_gen_101.sh</code></td>
</tr>
<tr>
  <td>CktBench-301</td>
  <td><code>bash scripts/train/baselines/cvaegan/cvaegan_cond_gen_301.sh</code></td>
</tr>
</table>

</details>

## 🙏 Acknowledgments

- We express our profound gratitude to **[Zehao Dong](https://scholar.google.com/citations?user=xcKId0oAAAAJ&hl=en), [Weidong Cao](https://sites.google.com/view/chalvescao/home), [Xuan Zhang](https://xzgroup.sites.northeastern.edu/our-team/), and the CktGNN team** for open-sourcing the OCB dataset and their pioneering work. Our research significantly benefited from the high-quality benchmarks and insights established in their work.
- Thanks to the authors of **PACE** for their foundational architectural research.

## 💬 Contact

- **Author**: Yuxuan Hou
- **Email**: yuxuan.hou.x@gmail.com
