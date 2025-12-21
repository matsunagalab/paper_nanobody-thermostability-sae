# Nanobody Thermostability Prediction using Supervised Fine-tuning and Sparse Autoencoders

## Description

This repository contains code for predicting nanobody thermostability using protein language models (pLMs). The project consists of two main components:

1. **Supervised Fine-tuning** (`supervised_finetuning/`): Code for fine-tuning pre-trained pLMs (e.g., ESM-2) on nanobody thermostability regression tasks.
2. **Sparse Autoencoder (SAE) Analysis** (`sparse_autoencoder/`): Code for interpreting fine-tuned pLMs using sparse autoencoders to extract and analyze interpretable features.

Each component requires a different Python environment, managed via Conda environment files (`env.yml`) located in their respective directories.

## Requirements

- Conda (or Miniconda/Mamba)
- Python 3.11
- CUDA-capable GPU (recommended for training and inference)
- Sufficient disk space for model checkpoints and activations

### Environment Setup

This project uses **two separate Conda environments** due to different dependency requirements:

1. **`esm` environment** (for supervised fine-tuning)
   - Environment file: `supervised_finetuning/env.yml`
   - Includes: PyTorch, Transformers, ESM, Optuna, PEFT (LoRA), etc.

2. **`interplm` environment** (for SAE analysis)
   - Environment file: `sparse_autoencoder/env.yml`
   - Includes: PyTorch, Transformers, nnsight, Streamlit, visualization libraries, etc.

**Important**: Always activate the appropriate environment before running code from each directory.

## Usage

### 1. Environment Setup

#### Step 1: Create the Supervised Fine-tuning Environment

```bash
cd supervised_finetuning
conda env create -f env.yml
conda activate esm
cd ..
```

#### Step 2: Create the SAE Analysis Environment

```bash
cd sparse_autoencoder
conda env create -f env.yml
conda activate interplm
pip install -e .  # Install the interplm package
cd ..
```

### 2. Running Supervised Fine-tuning

**Environment**: `esm` (activate with `conda activate esm`)  
**Directory**: `supervised_finetuning/`

The main script is `plm_supervised_fine_tuning.py`, which fine-tunes a pre-trained protein language model for regression tasks.

#### Required Arguments

- `--train_data_path`: Path to the training CSV file (must contain `sequence_aho_ungapped` and `tm` columns)
- `--val_data_path`: Path to the validation CSV file (same format as training data)
- `--model_path`: Hugging Face Hub model identifier or local path to the pre-trained base model
- `--tokenizer_path`: Path to the tokenizer
- `--output_dir`: Directory where the trained model and results will be saved

#### Common Optional Arguments

- `--embedding_type`: Method for aggregating token embeddings (`cls` or `mean`, default: `mean`)
- `--head_type`: Regression head architecture (`mlp` or `ridge`, default: `ridge`)
- `--num_train_epochs`: Number of training epochs (default: `50`)
- `--batch_size`: Batch size for training (default: `16`)
- `--max_length`: Maximum sequence length for tokenization (default: `256`)
- `--seed`: Random seed for reproducibility (default: `42`)

#### LoRA Fine-tuning

For parameter-efficient fine-tuning using LoRA:

- `--use_lora`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: `16`)
- `--lora_alpha`: LoRA scaling factor (default: `32`)
- `--lora_dropout`: LoRA dropout probability (default: `0.2`)

#### Hyperparameter Optimization

- `--optimize`: Enable Optuna-based hyperparameter optimization for encoder/LoRA parameters
- `--n_trials`: Number of Optuna trials (default: `100`)

#### Learning Rate and Regularization

- `--encoder_lr`: Learning rate for the encoder (default: `1e-4`)
- `--encoder_weight_decay`: Weight decay for the encoder (default: `0.01`)
- `--head_lr`: Learning rate for the regression head (default: `1e-3`)
- `--head_weight_decay`: Weight decay for the regression head (default: `0.001`)
- `--head_dropout_rate`: Dropout rate for the regression head (default: `0.2`)

#### Example Commands

**Basic fine-tuning without LoRA:**

```bash
conda activate esm
cd supervised_finetuning

python plm_supervised_fine_tuning.py \
    --train_data_path ../data/nbthermo/vhh_tm_dataset_train.csv \
    --val_data_path ../data/nbthermo/vhh_tm_dataset_validation.csv \
    --model_path facebook/esm2_t6_8M_UR50D \
    --tokenizer_path facebook/esm2_t6_8M_UR50D \
    --output_dir ./models/esm2_8m_finetuned \
    --embedding_type mean \
    --head_type ridge \
    --head_lr 0.1 \
    --head_weight_decay 0.01 \
    --head_lr 5e-05 \
    --head_weight_decay 0.0001 \
    --num_train_epochs 50 \
    --batch_size 16
```

**Fine-tuning with LoRA:**

```bash
conda activate esm
cd supervised_finetuning

python plm_supervised_fine_tuning.py \
    --train_data_path ../data/nbthermo/vhh_tm_dataset_train.csv \
    --val_data_path ../data/nbthermo/vhh_tm_dataset_validation.csv \
    --model_path facebook/esm2_t6_8M_UR50D \
    --tokenizer_path facebook/esm2_t6_8M_UR50D \
    --output_dir ./outputs/esm2_8m_lora \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 50 \
    --batch_size 16
```

**With hyperparameter optimization:**

```bash
conda activate esm
cd supervised_finetuning

python plm_supervised_fine_tuning.py \
    --train_data_path ../data/nbthermo/vhh_tm_dataset_train.csv \
    --val_data_path ../data/nbthermo/vhh_tm_dataset_validation.csv \
    --model_path facebook/esm2_t6_8M_UR50D \
    --tokenizer_path facebook/esm2_t6_8M_UR50D \
    --output_dir ./outputs/esm2_8m_optimized \
    --optimize \
    --n_trials 100 \
    --num_train_epochs 50 \
    --batch_size 16
```

**Note**: The script expects CSV files with columns `sequence_aho_ungapped` (protein sequences) and `tm` (thermostability target values).

### 3. Running SAE Analysis

**Environment**: `interplm` (activate with `conda activate interplm`)  
**Directory**: `sparse_autoencoder/`

After fine-tuning a pLM, you can use sparse autoencoders (SAEs) to interpret the model's learned features. The complete workflow consists of several steps:

1. Generate embeddings from FASTA files for SAE training
2. Train the SAE
3. Generate dense embeddings from FASTA files for evaluation
4. Convert dense embeddings to sparse representations using the trained SAE
5. Analyze sparse activation patterns

#### Step 1: Generate Embeddings for SAE Training

Use `fasta_to_sae_dataset.py` to generate embedding data from FASTA files for training the SAE.

**Arguments**:
- `--fasta_dir`: Directory containing FASTA files
- `--output_dir`: Directory to save embeddings
- `--esm_model_name`: ESM model name (e.g., `esm2_t6_8M_UR50D`)
- `--weight_file`: (Optional) Path to fine-tuned model weights (`.pt` format) to use fine-tuned model instead of pretrained
- `--disable_chunking`: Disable chunking and process all data at once
- `--start_shard`: First shard number to process (default: `0`)
- `--end_shard`: Last shard number to process (default: last shard)

**Example Commands**:

For pretrained model:
```bash
conda activate interplm
cd sparse_autoencoder

python interplm/esm/fasta_to_sae_dataset.py \
    --fasta_dir ../data/indi \
    --output_dir ./ngs/embeddings/pretrained_8m \
    --esm_model_name esm2_t6_8M_UR50D \
    --disable_chunking \
    --start_shard 0 \
    --end_shard 0
```

For fine-tuned model:
```bash
conda activate interplm
cd sparse_autoencoder

python interplm/esm/fasta_to_sae_dataset.py \
    --fasta_dir ../data/indi \
    --output_dir ./ngs/embeddings/sft_8m \
    --esm_model_name esm2_t6_8M_UR50D \
    --weight_file ../supervised_finetuning/models/esm2_8m_base-sft/encoder_lr_5e-5_batch_size_16_encoder_weight_decay_0.0001/encoder/converted_model.pt \
    --disable_chunking \
    --start_shard 0 \
    --end_shard 0
```

#### Step 2: Train the SAE

Use `train_plm_sae.py` to train a sparse autoencoder on the generated embeddings.

**Required Arguments**:
- `--plm_embd_dir`: Directory containing PLM embeddings (e.g., `./ngs/embeddings/pretrained_8m/layer_6`)
- `--save_dir`: Directory to save the trained SAE model

**Common Optional Arguments**:
- `--expansion_factor`: Dictionary size expansion factor relative to input dimension (default: `8`)
- `--batch_size`: Batch size for training (default: `32`)
- `--steps`: Total number of training steps (default: `100000`)
- `--lr`: Learning rate (default: `1e-3`)
- `--l1_penalty`: L1 regularization coefficient (default: `1e-1`)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: `50`)
- `--seed`: Random seed (default: `0`)
- `--eval_steps`: Frequency of evaluation steps (default: `1000`)
- `--save_steps`: Frequency of saving checkpoints (default: `50`)
- `--log_steps`: Frequency of logging (default: `100`)

**Example Commands**:

```bash
conda activate interplm
cd sparse_autoencoder

# Train SAE for pretrained model
python interplm/train/train_plm_sae.py \
    --plm_embd_dir ./ngs/embeddings/pretrained_8m/layer_6 \
    --save_dir ./models/pretrained_8m_100k/expansion_32_lr_9e-5_l1_5e-2/layer_6 \
    --expansion_factor 32 \
    --lr 9e-5 \
    --l1_penalty 5e-2

# Train SAE for fine-tuned model
python interplm/train/train_plm_sae.py \
    --plm_embd_dir ./ngs/embeddings/sft_8m/layer_6 \
    --save_dir ./models/sft_8m_100k/expansion_32_lr_9e-5_l1_5e-2/layer_6 \
    --expansion_factor 32 \
    --lr 9e-5 \
    --l1_penalty 5e-2
```

#### Step 3: Generate Dense Embeddings for Evaluation

Use `single_sequence_activations.py` to generate dense embeddings from FASTA files for evaluation datasets.

**Arguments**:
- `--fasta_file`: Path to FASTA file
- `--output_dir`: Directory to save embeddings
- `--esm_model_name`: ESM model name (default: `esm2_t6_8M_UR50D`)
- `--layers`: List of layer numbers to extract (default: `[1, 2, 3, 4, 5, 6]`)
- `--weight_file`: (Optional) Path to fine-tuned model weights to use fine-tuned model
- `--process_all`: Process all sequences in the FASTA file

**Example Commands**:

For pretrained model:
```bash
conda activate interplm
cd sparse_autoencoder

python interplm/esm/single_sequence_activations.py \
    --fasta_file ../data/nbthermo/vhh_tm_dataset.fasta \
    --output_dir ./interplm/nbthermo/nbthermo_embedding_esm_pretrained_8m \
    --layers 6 \
    --process_all
```

For fine-tuned model:
```bash
conda activate interplm
cd sparse_autoencoder

python interplm/esm/single_sequence_activations.py \
    --fasta_file ../data/nbthermo/vhh_tm_dataset.fasta \
    --output_dir ./interplm/nbthermo/nbthermo_embedding_esm_sft_8m \
    --weight_file ../supervised_finetuning/models/esm2_8m_base-sft/encoder_lr_5e-5_batch_size_16_encoder_weight_decay_0.0001/encoder/converted_model.pt \
    --layers 6 \
    --process_all
```

#### Step 4: Convert Dense Embeddings to Sparse Representations

Use `convert_to_sparse.py` to convert dense activations to sparse representations using the trained SAE.

**Required Arguments**:
- `--activation_dir`: Directory containing dense activation files
- `--output_dir`: Directory to save sparse activations
- `--sae_weight_file`: Path to trained SAE weight file (e.g., `./models/pretrained_8m_100k/expansion_32_lr_9e-5_l1_5e-2/layer_6/ae.pt`)
- `--target_layers`: List of layer numbers to convert to sparse representation
- `--all_layers`: List of all layer numbers to load (must include target_layers)

**Optional Arguments**:
- `--batch_size`: Batch size for processing (default: `1024`)
- `--device`: Device for computation (`auto`, `cpu`, or `cuda`, default: `auto`)

**Example Commands**:

```bash
conda activate interplm
cd sparse_autoencoder

# Convert for pretrained model
python interplm/esm/convert_to_sparse.py \
    --activation_dir ./interplm/nbthermo/nbthermo_embedding_esm_pretrained_8m \
    --output_dir ./interplm/nbthermo/nbthermo_embedding_sparse_pretrained_8m_expansion_32_lr_9e-5_l1_5e-2 \
    --target_layers 6 \
    --sae_weight_file ./models/pretrained_8m_100k/expansion_32_lr_9e-5_l1_5e-2/layer_6/ae.pt \
    --all_layers 6

# Convert for fine-tuned model
python interplm/esm/convert_to_sparse.py \
    --activation_dir ./interplm/nbthermo/nbthermo_embedding_esm_sft_8m \
    --output_dir ./interplm/nbthermo/nbthermo_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_5e-2 \
    --target_layers 6 \
    --sae_weight_file ./models/sft_8m_100k/expansion_32_lr_9e-5_l1_5e-2/layer_6/ae.pt \
    --all_layers 6
```

#### Step 5: Analyze Sparse Activation Patterns

Use `analyze_sae.py` to analyze sparse activation features for protein thermal stability prediction and generate an HTML report with visualizations.

**Arguments**:
- `--data-dir`: Directory containing dense activations
- `--sparse-dir`: Directory containing sparse activations
- `--tm-data`: Path to TM dataset CSV file
- `--output-dir`: Output directory for results (default: `./output`)
- `--layer`: Layer number for activations (default: `6`)
- `--test-size`: Test set size for train/test split (default: `0.2`)
- `--random-state`: Random state for reproducibility (default: `42`)
- `--cv-folds`: Number of CV folds for RidgeCV (default: `10`)
- `--feature-indices`: Feature indices to analyze in detail (default: `[1688, 1677]`)

**Example Commands**:

```bash
conda activate interplm
cd sparse_autoencoder

# Analyze pretrained model
python analyze_sae.py \
    --data-dir ./interplm/nbthermo/nbthermo_embedding_esm_pretrained_8m \
    --sparse-dir ./interplm/nbthermo/nbthermo_embedding_sparse_pretrained_8m_expansion_32_lr_9e-5_l1_5e-2 \
    --tm-data ../data/nbthermo/vhh_tm_dataset.csv \
    --output-dir ./outputs/pretrained_8m_100k_expansion_32_lr_9e-5_l1_5e-2

# Analyze fine-tuned model
python analyze_sae.py \
    --data-dir ./interplm/nbthermo/nbthermo_embedding_esm_sft_8m \
    --sparse-dir ./interplm/nbthermo/nbthermo_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_5e-2 \
    --tm-data ../data/nbthermo/vhh_tm_dataset.csv \
    --output-dir ./outputs/sft_8m_100k_expansion_32_lr_9e-5_l1_5e-2
```

#### Additional Tools

The `sparse_autoencoder/interplm/` package contains various modules for:

- **Feature visualization**: `interplm/feature_vis/`
- **Interactive dashboard**: `interplm/dashboard/app.py` (Streamlit-based)
- **Concept analysis**: `interplm/concept/`
- **Data processing**: `interplm/data_processing/`

Refer to the `sparse_autoencoder/README.md` for more detailed information about the InterPLM toolkit.

### Important Notes

1. **Environment Switching**: Always ensure you have activated the correct environment before running scripts:
   - Use `conda activate esm` for supervised fine-tuning
   - Use `conda activate interplm` for SAE analysis

2. **Working Directory**: Make sure you are in the correct directory when running scripts, or adjust paths accordingly.

3. **Data Format**: Training/validation CSV files must contain:
   - `sequence_aho_ungapped`: Protein sequence strings
   - `tm`: Numeric thermostability values (target variable)

4. **Output Structure**: Fine-tuned models are saved with separate directories for the encoder and regression head. When using LoRA, adapters are saved relative to the base model for easy reloading.
