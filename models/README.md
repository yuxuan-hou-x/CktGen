# Models Documentation

Comprehensive documentation of all models, architectures, and components in CktGen.

## Table of Contents

- [Overview](#overview)
- [Model Types](#model-types)
- [Encoder Architectures](#encoder-architectures)
- [Model Components](#model-components)
- [Loss Functions](#loss-functions)
- [Model Selection Guide](#model-selection-guide)
- [Code Structure](#code-structure)

---

## Overview

CktGen provides a modular framework with:

- **5 Main Model Types**: VAE, CktGen (CVAE), LDT (Latent Diffusion), CVAE-GAN, Evaluator
- **4 Encoder Architectures**: CktArchi, PACE, CktGNN, DIGIN
- **6 Loss Functions**: Reconstruction, KL, Alignment, Contrastive, Guidance, Prediction

Models are specified via the `--modelname` parameter:

```
--modelname [modeltype]_[architecture]_[loss1]_[loss2]_...
```

Examples:

```bash
--modelname vae_cktarchi_recon_kl
--modelname cktgen_pace_kl_recon_align_nce_gde
--modelname ldt_cktarchi_mse
```

---

## Model Types

### 1. VAE (Variational Autoencoder)

**Purpose**: Random circuit generation, learning latent representations

**Architecture**:

```
Circuit → Encoder → μ, log_σ² → Latent (z) → Decoder → Circuit'
                      ↓
                 Reparameterization
```

**Key Features**:

- Unsupervised learning
- Smooth latent space for interpolation
- Prior sampling for generation

**Use Cases**:

- Circuit reconstruction
- Random circuit generation
- Latent space analysis
- Pretraining for LDT

**Training**:

```bash
python train/train_vae.py \
  --modelname vae_cktarchi \
  --vae \
  --lambda_kl 5e-3 \
  --lambda_recon 1.0
```

**Implementation**: `models/modeltype/vae.py`

---

### 2. CktGen (Conditional VAE)

**Purpose**: Specification-driven circuit generation

**Architecture**:

```
Circuit → Circuit_Encoder ─────┐
                               ├→ Latent (z) → Decoder → Circuit'
Spec → Spec_Encoder ───────────┘
           │
           └──────────→ Cross-Modal Alignment
           └──────────→ Contrastive Learning
           └──────────→ Classifier-Free Guidance
```

**Key Features**:

- **Cross-modal alignment**: Aligns circuit and spec embeddings
- **Contrastive learning**: InfoNCE loss for discrimination
- **Filtered sampling**: Removes false negatives based on spec similarity
- **Classifier-free guidance**: Enhances conditioning during inference

**Use Cases**:

- Inverse design (spec → circuit)
- Conditional generation
- Circuit optimization

**Training**:

```bash
python train/train_cktgen.py \
  --modelname cktgen_cktarchi_kl_recon_align_nce_gde \
  --vae --conditioned --contrastive --guided --filter \
  --lambda_kl 1e-5 \
  --lambda_align 1.0 \
  --lambda_nce 1.0 \
  --temperature 0.1
```

**Implementation**: `models/modeltype/cktgen.py`

---

### 3. LDT (Latent Diffusion Transformer)

**Purpose**: State-of-the-art generation quality

**Architecture**:

```
Circuit → VAE_Encoder → z₀
Spec ────────────────────┐
                         ↓
z₀ + noise → Denoiser (Transformer) → z₀' → VAE_Decoder → Circuit'
   (zₜ)         (predicts noise)
```

**Key Features**:

- Diffusion in latent space (faster than pixel space)
- Transformer-based denoiser
- Iterative refinement
- High-quality generation

**Use Cases**:

- High-fidelity circuit generation
- Complex specification matching
- Research benchmarking

**Training**:

```bash
# Step 1: Train VAE
python train/train_vae.py --modelname vae_cktarchi ...

# Step 2: Train LDT
python train/train_ldt.py \
  --modelname ldt_cktarchi_mse \
  --vae_pth ./output/vae_checkpoint.pth \
  --conditioned
```

**Implementation**: `models/modeltype/ldt.py`

---

### 4. CVAEGAN (Conditional VAE-GAN)

**Purpose**: Adversarial training for sharper generation

**Architecture**:

```
Circuit + Spec → Encoder → z → Decoder → Circuit'
                                   ↓
                            Discriminator → Real/Fake
```

**Key Features**:

- Adversarial loss for realism
- Conditional on specifications
- Can produce sharper outputs than VAE

**Training**:

```bash
python train/train_cvaegan.py \
  --modelname cvaegan_cktarchi \
  --conditioned
```

**Implementation**: `models/modeltype/cvaegan.py`

---

### 5. Evaluator (Performance Predictor)

**Purpose**: Predict circuit performance from topology

**Architecture**:

```
Circuit → Encoder → Fully Connected → Performance Specs
```

**Use Cases**:

- Fast performance estimation
- Guidance for generation
- Fitness evaluation in optimization

**Training**:

```bash
python train/train_evaluator.py \
  --modelname evaluator_cktarchi_pred \
  --lambda_pred 1.0
```

**Implementation**: `models/modeltype/evaluator.py`

---

## Encoder Architectures

### 1. CktArchi (Transformer-based)

**Best for**: High-quality generation, complex circuits

**Architecture**:

```
Graph → Node Embedding + Positional Encoding
     → Multi-Head Self-Attention (x N layers)
     → Feed-Forward Networks
     → Global Pooling
     → Latent Vector
```

**Parameters**:

```python
{
    'emb_dim': 128,          # Embedding dimension
    'hidden_dim': 256,       # Hidden layer size
    'latent_dim': 64,        # Latent vector size
    'num_layers': 6,         # Transformer layers
    'num_heads': 8,          # Attention heads
    'dropout_rate': 0.3,     # Dropout rate
}
```

**Pros**:

- ✓ Best generation quality
- ✓ Handles complex topologies
- ✓ Attention mechanism captures long-range dependencies

**Cons**:

- ✗ Slower than GNN-based
- ✗ More parameters

**Usage**:

```bash
--modelname [model]_cktarchi_[losses]
```

**Implementation**: `models/architectures/cktarchi.py`

---

### 2. PACE (Parallel Architecture with Convolution Encoder)

**Best for**: Large circuits, faster training

**Architecture**:

```
Graph → Graph Convolution (x N layers)
     → Node-level and Graph-level Features
     → Concatenate
     → Latent Vector
```

**Parameters**:

```python
{
    'emb_dim': 128,
    'hidden_dim': 256,
    'latent_dim': 64,
    'num_layers': 4,
    'num_heads': 4,
}
```

**Pros**:

- ✓ Faster than CktArchi
- ✓ Good for large graphs
- ✓ Fewer parameters

**Cons**:

- ✗ Slightly lower quality than CktArchi

**Usage**:

```bash
--modelname [model]_pace_[losses]
```

**Implementation**: `models/architectures/pace.py`

---

### 3. CktGNN (Graph Neural Network)

**Best for**: Graph-structured reasoning

**Architecture**:

```
Graph → GIN Layers (Graph Isomorphism Network)
     → Message Passing
     → Graph Pooling
     → Latent Vector
```

**Parameters**:

```python
{
    'emb_dim': 128,
    'hid_dim': 256,
    'latent_dim': 64,
    'bidirectional': True,
}
```

**Pros**:

- ✓ Explicit graph structure modeling
- ✓ Interpretable message passing
- ✓ Good for small-medium circuits

**Cons**:

- ✗ Can struggle with very large graphs

**Usage**:

```bash
--modelname [model]_cktgnn_[losses]
```

**Implementation**: `models/architectures/cktgnn.py`

---

### 4. DIGIN (Lightweight)

**Best for**: Fast experiments, resource-constrained settings

**Architecture**:

```
Graph → Simple GNN
     → Lightweight Pooling
     → Latent Vector
```

**Parameters**:

```python
{
    'emb_dim': 64,
    'hidden_dim': 128,
    'latent_dim': 32,
}
```

**Pros**:

- ✓ Very fast training and inference
- ✓ Low memory usage
- ✓ Good for quick prototyping

**Cons**:

- ✗ Lower quality than other architectures

**Usage**:

```bash
--modelname [model]_digin_[losses]
```

**Implementation**: `models/architectures/digin.py`

---

## Model Components

### Encoder

Converts circuit graph to latent vector:

```python
z = encoder(circuit_graph)  # z.shape: [batch, latent_dim]
```

**Implementations**:

- CktArchi: `models/architectures/ckt_encoder.py`
- Others: Integrated in architecture files

---

### Decoder

Reconstructs circuit from latent vector:

```python
circuit' = decoder(z)
```

**Autoregressive generation**:

```python
# Generate node by node
for t in range(max_nodes):
    node_type[t], node_size[t] = decoder.step(z, context)
    connections[t] = decoder.connect(z, node_type[:t])
```

**Implementation**: `models/architectures/ckt_decoder.py`

---

### Spec Encoder

Encodes performance specifications:

```python
spec_emb = spec_encoder(specs)  # specs: [batch, num_specs]
```

**Architecture**:

```
Specs → FC Layer → ReLU → FC Layer → Embedding
```

**Implementation**: `models/architectures/spec_encoder.py`

---

### Denoiser (for LDT)

Predicts noise in diffusion process:

```python
noise_pred = denoiser(z_t, t, spec)
z_{t-1} = scheduler.step(z_t, noise_pred, t)
```

**Architecture**: Transformer with timestep embedding

**Implementation**: `models/modeltype/ldt_module/denoiser.py`

---

## Loss Functions

### 1. Reconstruction Loss (`lambda_recon`)

**Purpose**: Ensure decoder can reconstruct input

**Formula**:

```
L_recon = CrossEntropy(predicted_nodes, true_nodes)
        + CrossEntropy(predicted_edges, true_edges)
        + MSE(predicted_sizes, true_sizes)
```

**Typical weight**: 1.0 (baseline)

**Use in**: VAE, CktGen, CVAEGAN

---

### 2. KL Divergence Loss (`lambda_kl`)

**Purpose**: Regularize latent space to be Gaussian

**Formula**:

```
L_kl = -0.5 * Σ(1 + log_σ² - μ² - σ²)
```

**Typical weights**:

- VAE: 5e-3 (stronger regularization)
- CktGen: 1e-5 (weaker, allows more flexibility)

**Use in**: VAE, CktGen

---

### 3. Alignment Loss (`lambda_align`)

**Purpose**: Align circuit and spec embeddings

**Formula**:

```
L_align = MSE(circuit_embedding, spec_embedding)
```

**Typical weight**: 1.0

**Use in**: CktGen (conditional models)

---

### 4. Contrastive Loss (`lambda_nce`)

**Purpose**: Discriminate between matching and non-matching circuit-spec pairs

**Formula** (InfoNCE):

```
L_nce = -log( exp(sim(c_i, s_i) / τ) / Σ_j exp(sim(c_i, s_j) / τ) )
```

where `τ` is temperature (`--temperature`)

**Typical weights**:

- `lambda_nce`: 1.0
- `temperature`: 0.1

**Use in**: CktGen with `--contrastive`

**With filtering** (`--filter`):

- Removes false negatives (circuits with similar specs)
- Improves contrastive learning quality

---

### 5. Guidance Loss (`lambda_gde`)

**Purpose**: Classifier-free guidance for better conditioning

**Formula**:

```
L_gde = MSE(conditional_output, unconditional_output)
```

**Typical weight**: 1.0

**Use in**: CktGen with `--guided`

**Inference**:

```
z' = z_uncond + guidance_scale * (z_cond - z_uncond)
```

---

### 6. Prediction Loss (`lambda_pred`)

**Purpose**: Train evaluator to predict specifications

**Formula**:

```
L_pred = MSE(predicted_specs, true_specs)
```

**Typical weight**: 1.0

**Use in**: Evaluator

---

## Model Selection Guide

### By Task

| Task                       | Recommended Model | Rationale                              |
| -------------------------- | ----------------- | -------------------------------------- |
| **Reconstruction**         | VAE + CktArchi    | Best quality, smooth latent space      |
| **Random Generation**      | VAE + CktArchi    | Simple, effective                      |
| **Conditional Generation** | CktGen + CktArchi | Multi-task learning, alignment         |
| **Best Quality**           | LDT + CktArchi    | State-of-the-art, iterative refinement |
| **Fast Inference**         | VAE + DIGIN       | Lightweight, single forward pass       |
| **Large Circuits**         | CktGen + PACE     | Handles complexity well                |

---

### By Dataset

| Dataset            | Recommended        | Rationale                         |
| ------------------ | ------------------ | --------------------------------- |
| **CktBench-101**   | CktArchi or CktGNN | Moderate complexity, best quality |
| **CktBench-301**   | PACE or CktArchi   | Handles large graphs              |
| **Custom (small)** | DIGIN              | Fast experiments                  |

---

### By Resources

| GPU Memory | Model      | Architecture | Batch Size |
| ---------- | ---------- | ------------ | ---------- |
| **8GB**    | VAE        | DIGIN        | 8          |
| **12GB**   | VAE/CktGen | CktGNN       | 16         |
| **24GB**   | CktGen/LDT | CktArchi     | 32         |
| **48GB**   | LDT        | CktArchi     | 64         |

---

## Code Structure

```
models/
├── get_model.py                # Model factory
├── architectures/              # Encoder/decoder implementations
│   ├── cktarchi.py
│   ├── pace.py
│   ├── cktgnn.py
│   ├── digin.py
│   ├── ckt_encoder.py         # CktArchi encoder
│   ├── ckt_decoder.py         # CktArchi decoder
│   ├── spec_encoder.py        # Spec encoder
│   └── tools/
│       └── gin.py             # GIN layers
├── modeltype/                  # Full model implementations
│   ├── vae.py
│   ├── cktgen.py
│   ├── ldt.py
│   ├── cvaegan.py
│   ├── evaluator.py
│   └── ldt_module/            # LDT components
│       ├── denoiser.py
│       ├── transformer_diffusion.py
│       ├── timestep_embedding.py
│       └── pos_encoding.py
└── tools/
    └── losses.py              # Loss functions
```

---

## Advanced Configurations

### Multi-Loss Training

Combine multiple losses:

```bash
--modelname cktgen_cktarchi_kl_recon_align_nce_gde \
--lambda_kl 1e-5 \
--lambda_recon 1.0 \
--lambda_align 1.0 \
--lambda_nce 1.0 \
--lambda_gde 1.0 \
--temperature 0.1
```

### Architecture Customization

Override default architecture params:

```bash
--modelname cktgen_cktarchi_recon_kl \
--num_layers 8 \        # Deeper network
--num_heads 16 \        # More attention heads
--latent_dim 128 \      # Larger latent space
--dropout_rate 0.2      # Less dropout
```

---

## Model Checkpoints

Checkpoints are saved as:

```
{out_dir}/{exp_name}/{modelname}_checkpoint{epoch}.pth
```

**Contents**:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'args': args,
    'train_loss': train_loss,
    'valid_loss': valid_loss,
}
```

**Loading**:

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Citation

For model-specific citations, see individual papers:

- **CktArchi**: CktGen paper
- **PACE**: [PACE Paper](https://arxiv.org/abs/2303.12024)
- **CktGNN**: [CktGNN Paper](https://arxiv.org/abs/2308.16406)
