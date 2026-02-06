# Options Parameters Analysis and Documentation Update

## Summary

This document summarizes the analysis of all command-line arguments in the `options/` directory, including which parameters are actually used in training, which were removed, and the rationale behind each decision.

## Files Updated

1. `options/base.py` - Miscellaneous and CUDA options
2. `options/dataset.py` - Dataset configuration options
3. `options/evaluation.py` - Evaluation and checkpoint options
4. `options/training.py` - Training hyperparameter options
5. `options/models.py` - Model architecture and loss function options

---

## Parameters Analysis

### ✅ Kept Parameters (Actually Used in Training)

#### base.py
- **`--out_dir`**: Used to create output directory for checkpoints and logs
- **`--exp_name`**: Creates subdirectory structure for organizing experiments
- **`--cuda/--cpu`**: Controls device selection (GPU/CPU)
- **`--print_iter`**: Controls logging frequency (currently logging code is commented out but parameter preserved for future use)

---

#### dataset.py
- **`--data_fold_name`**: Used to locate dataset directory (CktBench101/301)
- **`--data_type`**: Dataset type identifier (OCB)
- **`--data_name`**: Pickle file name for loading circuit graphs
- **`--graph_numbers`**: Number of circuits to process for visualization and statistical analysis (not used in training)

---

#### evaluation.py
- **`--eval_interval`**: Controls evaluation frequency during training (every N epochs)
- **`--resume_pth`**: Loads model checkpoint for resuming training or evaluation
- **`--pretrained_eval_resume_pth`**: Loads evaluator model for computing metrics
- **`--vae_pth`**: Loads pretrained VAE for LDT training
- **`--infer_batch_size`**: Batch size for inference and evaluation (reconstruction, prior validity, generation)

**Removed**: 
- `--cktgen_pth` (defined but never used anywhere in the codebase)

---

#### training.py
All parameters are actively used:
- **`--lr`**: AdamW learning rate
- **`--epochs`**: Maximum training epochs
- **`--batch_size`**: Training batch size
- **`--seed`**: Random seed for reproducibility
- **`--beta`**: AdamW beta parameters [beta1, beta2]
- **`--weight_decay`**: L2 regularization coefficient
- **`--save_interval`**: Checkpoint saving frequency

---

#### models.py

##### Architecture-Specific Parameters (All Used)

**CKTGNN** (`add_cktgnn_options`):
- `--max_n`, `--max_pos`, `--num_types`, `--emb_dim`, `--feat_emb_dim`, `--hid_dim`, `--latent_dim`, `--bidirectional`, `--sized`

**CktArchi** (`add_cktarchi_options`):
- `--max_n`, `--emb_dim`, `--hidden_dim`, `--latent_dim`, `--size_emb_dim`, `--ff_size`, `--num_layers`, `--num_heads`, `--num_types`, `--num_paths`, `--block_size`, `--dropout_rate`, `--fc_rate`, `--type_rate`, `--path_rate`, `--size_rate`

**DIGIN** (`add_digin_options`):
- `--max_n`, `--emb_dim`, `--hidden_dim`, `--latent_dim`, `--size_emb_dim`, `--num_types`, `--num_paths`, `--dropout`

**PACE** (`add_pace_options`):
- `--num_types`, `--emb_dim`, `--v_size_emb_dim`, `--hidden_dim`, `--num_heads`, `--num_layers`, `--dropout`, `--fc_hidden`, `--latent_dim`

##### General Model Parameters

**Used**:
- **`--modelname`**: Specifies model type, architecture, and losses
- **`--conditioned`**: Enables conditional generation
- **`--vae`**: Enables variational autoencoder
- **`--eps_factor`**: VAE reparameterization noise scaling (used in CKTGNN, PACE, CVAEGAN)
- **`--contrastive`**: Enables InfoNCE contrastive learning
- **`--guided`**: Enables classifier-free guidance
- **`--filter`**: Enables filtered contrastive learning (masks false negatives)
- **`--lambda_recon`**: Weight for reconstruction loss
- **`--lambda_kl`**: Weight for KL divergence loss
- **`--lambda_align`**: Weight for cross-modal alignment loss
- **`--lambda_gde`**: Weight for guidance loss
- **`--lambda_nce`**: Weight for InfoNCE contrastive loss
- **`--lambda_pred`**: Weight for spec prediction loss (evaluator only)
- **`--temperature`**: Temperature for InfoNCE loss

**Removed**:
- `--lambda_mse` - Defined but never used in any loss computation
- `--vae_path` - Type error (should be str not float), never used

---

## Usage Examples

### Train CktGen on CktBench101
```bash
python train/train_cktgen.py \
  --modelname cktgen_cktarchi_kl_recon_align_nce_gde \
  --data_fold_name CktBench101 \
  --data_name ckt_bench_101 \
  --exp_name cktgen/OCB101/filter/KL_1e-5_T_1e-1 \
  --vae --conditioned --contrastive --guided --filter \
  --lambda_kl 1e-5 \
  --lambda_nce 1.0 \
  --lambda_align 1.0 \
  --temperature 0.1 \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 1000 \
  --eval_interval 30 \
  --save_interval 100
```

### Train VAE (Reconstruction Only)
```bash
python train/train_vae.py \
  --modelname vae_cktarchi_kl_recon \
  --data_fold_name CktBench101 \
  --vae \
  --lambda_kl 5e-3 \
  --batch_size 32 \
  --epochs 600
```

### Train LDT (Latent Diffusion)
```bash
python train/train_ldt.py \
  --modelname ldt_cktarchi \
  --data_fold_name CktBench101 \
  --vae_pth ./output/vae_checkpoint600.pth \
  --batch_size 32 \
  --epochs 1000
```

---

## Parameter Categories

### 1. Essential Parameters (Always Needed)
- `modelname`, `data_fold_name`, `data_name`, `out_dir`, `exp_name`

### 2. Architecture Parameters (Model-Specific)
- Different for each architecture (cktarchi, pace, cktgnn, digin)
- Automatically loaded based on `modelname`

### 3. Training Hyperparameters
- `lr`, `batch_size`, `epochs`, `seed`, `beta`, `weight_decay`
- Generally use defaults, tune `lr` and `batch_size` if needed

### 4. Loss Weights (Task-Specific)
- VAE: `lambda_kl`, `lambda_recon`
- Conditional: Add `lambda_align`
- Contrastive: Add `lambda_nce`, `temperature`
- Guidance: Add `lambda_gde`

### 5. Checkpoint & Evaluation
- `save_interval`, `eval_interval`, `resume_pth`, `pretrained_eval_resume_pth`, `vae_pth`, `infer_batch_size`

### 6. Logging & Visualization
- `print_iter`: Training progress logging frequency (currently commented out but available)
- `graph_numbers`: Number of circuits for visualization and statistical analysis

---

## Changes Made

### Removed Parameters
1. **`--cktgen_pth`** (evaluation.py): Never referenced anywhere in the codebase
2. **`--lambda_mse`** (models.py): Defined but not used in any loss function
3. **`--vae_path`** (models.py): Type error (defined as float instead of str) and never used

### Enhanced Documentation
- Added detailed docstrings to all parameter groups
- Explained the purpose and typical values for each parameter
- Provided context about when parameters are used
- Included examples and relationships between parameters
- Clarified model type, architecture, and loss function naming conventions

---

## Verification

All changes verified by:
1. ✅ Searching codebase for actual parameter usage
2. ✅ Checking training scripts (train_cktgen.py, train_vae.py, train_ldt.py, etc.)
3. ✅ Verifying model initialization in get_model.py
4. ✅ Analyzing loss computation in model classes
5. ✅ Running linter - no errors introduced
6. ✅ All shell scripts in `scripts/` still reference valid parameters

---

## Recommendations for Future Use

1. **For new experiments**: Use the shell scripts in `scripts/` as templates
2. **For debugging**: Start with smaller `batch_size` and higher `lambda_kl`
3. **For best performance**: Use CktArchi architecture with filter=True
4. **For faster training**: Reduce `num_layers` or use DIGIN/PACE architectures
5. **For better generation**: Enable `--contrastive --guided --filter` with appropriate loss weights

