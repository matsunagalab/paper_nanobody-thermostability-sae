#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8
"""
A script to fine-tune a pre-trained model for protein sequence regression tasks.

Overview:
1.  **Data Loading**: Loads a CSV file containing protein sequences and their corresponding
    target variables (e.g., thermal stability 'tm').
2.  **Model Construction**:
    -   Loads a pre-trained Transformer model (e.g., ESM-2) from the Hugging Face Hub
        as the base encoder.
    -   Adds a regression head of a specified type (MLP or Ridge)
        on top of the base model.
3.  **Fine-tuning Strategy**:
    -   **Full Fine-tuning**: Updates all parameters of the model (default).
    -   **LoRA (Low-Rank Adaptation)**: By using the `--use_lora` flag, switches to
        parameter-efficient fine-tuning via the PEFT library. In this mode, the
        base model's weights are frozen, and only small adapter layers are trained.
4.  **Hyperparameter Optimization (Optional)**:
    -   If the `--optimize` flag is specified, it uses Optuna to search for the
        optimal hyperparameters for the encoder (lr, weight_decay) and LoRA (r, alpha, dropout).
        The head parameters are fixed based on command-line arguments or defaults.
    -   If not optimizing, it uses values from command-line arguments or defaults.
5.  **Training and Evaluation**:
    -   Trains the final model using the determined hyperparameters.
    -   Evaluates performance on the evaluation dataset after each epoch and saves
        the best-performing model.
6.  **Model Saving**:
    -   Saves the trained model components (transformer backbone/LoRA adapters,
        regression head), the tokenizer, and the hyperparameters used to a specified
        output directory.
    -   If LoRA was used, the adapter is saved relative to the base model, creating a
        self-contained and easily reloadable directory.
    -   If Optuna was used, the best parameters for the encoder are saved to a JSON file.

Command-line Arguments:
    --train_data_path (str, required):
        Path to the CSV file for training.
    --val_data_path (str, required):
        Path to the CSV file for validation.
    --model_path (str, required):
        Name on the Hugging Face Hub or local path to the pre-trained base model.
    --tokenizer_path (str, required):
        Path to the tokenizer.
    --output_dir (str, required):
        Path to the directory where the trained model and results will be saved.
    --embedding_type (str):
        Method for aggregating token embeddings from the transformer.
        Choices: 'cls' or 'mean'. Default: "mean"
    --head_type (str):
        Architecture for the regression head.
        Choices: 'mlp', 'ridge'. Default: "mlp"
    --num_train_epochs (int):
        Number of training epochs. Default: 50
    --batch_size (int):
        Batch size for training. Default: 16
    --optimize (bool):
        If set, enables hyperparameter optimization for encoder/LoRA. Default: False
    --n_trials (int):
        Number of trials for Optuna to run. Used only if `--optimize` is enabled. Default: 100
    --max_length (int):
        Maximum sequence length for tokenization. Longer sequences will be truncated. Default: 256
    --seed (int):
        Random seed for reproducibility. Default: 42
    --use_lora (bool):
        If set, enables parameter-efficient fine-tuning using LoRA. Default: False
    --lora_r (int):
        The rank (r) of the LoRA update matrices. Default: 16
    --lora_alpha (int):
        The LoRA scaling factor (alpha). Default: 32
    --lora_dropout (float):
        The dropout probability for LoRA layers. Default: 0.2
    
    # Head hyperparameters
    --head_lr (float):
        Learning rate for the regression head. Default: 1e-3
    --head_weight_decay (float):
        Weight decay for the regression head. Default: 0.0
    --head_dropout_rate (float):
        Dropout rate for the regression head. Default: 0.2
    --head_activate_fnc (str):
        Activation function for the regression head. Default: "ReLU"
    
    # Encoder hyperparameters
    --encoder_lr (float):
        Learning rate for the encoder. Default: 1e-4
    --encoder_weight_decay (float):
        Weight decay for the encoder. Default: 0.01
"""
import os
import json
import random
import argparse
import gc
import glob
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from safetensors.torch import save_file

# (LORA_CONFIG, RegressionModel, CustomTrainerなどのクラス定義は変更なし)
# --- Optional PEFT / LoRA import ------------------------------------------------
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:  # PEFT is optional – only needed when using LoRA
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None

# --- LoRA target_modules helper -------------------------------------------------
def get_lora_config(args, model):
    if not hasattr(model, 'state_dict'):
        raise ValueError("Model must be a torch.nn.Module instance")

    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("encoder" in name or "esm" in name):
            target_modules.append(name.split(".")[-1])

    target_modules = list(set(target_modules))
    if not target_modules:
        raise ValueError("No suitable Linear layers found for LoRA adaptation.")

    print(f"[LoRA] Applying to modules: {target_modules}")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules
    )


class RegressionModel(torch.nn.Module):
    """Encoder + regression head.

    If `base_model` is a PeftModel (i.e. LoRA‑wrapped), only the injected LoRA
    parameters will be trainable while the underlying backbone remains frozen.
    """

    def __init__(self, base_model, dropout=0.2, act_fn=torch.nn.ReLU(), emb_type="mean", head_type="mlp"):
        super().__init__()
        self.encoder = base_model
        hidden_size = base_model.config.hidden_size
        self.emb_type = emb_type
        self.head_type = head_type

        # Build the regression head
        if head_type == "mlp":
            d1 = max(hidden_size // 2, 1)
            d2 = max(hidden_size // 4, 1)
            d3 = max(hidden_size // 16, 1)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, d1),
                act_fn,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(d1, d2),
                act_fn,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(d2, d3),
                act_fn,
                torch.nn.Linear(d3, 1),
            )
        elif head_type == "ridge":
            self.head = torch.nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Embedding aggregation
        if self.emb_type == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            mask = attention_mask.float().clone()
            mask[:, 0] = 0  # CLS
            eos_indices = attention_mask.sum(1) - 1
            mask.scatter_(1, eos_indices.unsqueeze(1), 0)
            emb_sum = (last_hidden * mask.unsqueeze(-1)).sum(1)
            token_count = mask.sum(1, keepdim=True).clamp(min=1e-9)
            pooled = emb_sum / token_count

        preds = self.head(pooled).view(-1)  # (batch,)

        if labels is not None:
            loss = F.mse_loss(preds, labels.view(-1))
            return SequenceClassifierOutput(loss=loss, logits=preds)
        else:
            return SequenceClassifierOutput(logits=preds)


class CustomTrainer(Trainer):
    def __init__(self, *args, head_params=None, encoder_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_params = head_params if head_params is not None else {}
        self.encoder_params = encoder_params if encoder_params is not None else {}
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)  # labels は inputs 内に含まれている
        loss = outputs.loss if hasattr(outputs,'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            encoder_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
            head_params = [p for n, p in self.model.named_parameters() if n.startswith("head.") and p.requires_grad]

            # For Ridge, weight_decay on head params acts as L2 regularization
            head_weight_decay = self.head_params.get("weight_decay", 0.0) if self.model.head_type == "ridge" else 0.0

            optimizer_grouped_parameters = [
                {
                    "params": encoder_params,
                    "lr": self.encoder_params.get("lr", self.args.learning_rate),
                    "weight_decay": self.encoder_params.get("weight_decay", self.args.weight_decay),
                },
                {
                    "params": head_params,
                    "lr": self.head_params.get("lr", self.args.learning_rate),
                    "weight_decay": head_weight_decay,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


# --- Metrics --------------------------------------------------------------------

def compute_metrics(pred):
    labels = pred.label_ids.reshape(-1)
    preds  = pred.predictions.reshape(-1)
    mse = mean_squared_error(labels, preds)
    return {"mse":mse, "rmse":np.sqrt(mse), "mae":mean_absolute_error(labels,preds), "r2":r2_score(labels,preds)}

# --- Build model ----------------------------------------------------------------

def build_model(base_model, head_params, args):
    act_fn_name = head_params.get("activate_fnc", "ReLU")
    dropout_rate = head_params.get("dropout_rate", 0.2)
    act_fn = getattr(torch.nn, act_fn_name)()
    return RegressionModel(
        base_model,
        dropout=dropout_rate,
        act_fn=act_fn,
        emb_type=args.embedding_type,
        head_type=args.head_type
    )

# --- Optuna objective -----------------------------------------------------------

def objective(trial, args, train_ds, eval_ds, head_params, tokenizer):
    # Parameters to be tuned by Optuna
    encoder_params = {
        "lr": trial.suggest_float("encoder_lr", 1e-5, 1e-3),
        "weight_decay": trial.suggest_float("encoder_weight_decay", 1e-4, 1e-1),
    }

    base_model = AutoModel.from_pretrained(args.model_path)
    trial_args = deepcopy(args)

    if args.use_lora:
        if get_peft_model is None:
            raise ImportError("peft is not installed but --use_lora was passed.")
        
        # Tune LoRA parameters
        trial_args.lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
        trial_args.lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
        trial_args.lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.5)
        
        lora_cfg = get_lora_config(trial_args, base_model)
        base_model = get_peft_model(base_model, lora_cfg)
        base_model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(base_model, head_params, trial_args).to(device)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/trial_{trial.number}",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,  # Dummy value, will be overridden by CustomTrainer
        weight_decay=0.0,    # Dummy value, will be overridden by CustomTrainer
        load_best_model_at_end=False,  # Set to False to keep all checkpoints during training
        metric_for_best_model="rmse",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=None,  # Keep all checkpoints during training
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        head_params=head_params,
        encoder_params=encoder_params,
    )   

    eval_dataloader = trainer.get_eval_dataloader(eval_ds)
    print(f"[DEBUG] Validation Dataset size: {len(eval_ds)}")
    print(f"[DEBUG] Evaluation DataLoader number of batches: {len(eval_dataloader)}")

    # Train the model
    train_result = trainer.train()
    
    # Evaluate the model
    result = trainer.evaluate()

    # Clean up trial directory completely
    if os.path.exists(trial_checkpoint_dir):
        shutil.rmtree(trial_checkpoint_dir)
        print(f"[Trial {trial.number}] Removed trial directory: {trial_checkpoint_dir}")

    del model, trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.get("eval_loss", float("inf"))


# --- Final training & saving ----------------------------------------------------

def train_final(params, args, train_ds, eval_ds, tokenizer):
    print("--- Starting final model training ---")
    encoder_params = params.get("encoder", {})
    head_params = params.get("head", {})
    print("Encoder params:", json.dumps(encoder_params, indent=2))
    print("Head params:", json.dumps(head_params, indent=2))

    base_model = AutoModel.from_pretrained(args.model_path)
    final_args = deepcopy(args)

    if args.use_lora:
        # If LoRA params were optimized or loaded, use them
        final_args.lora_r = params.get("lora_r", args.lora_r)
        final_args.lora_alpha = params.get("lora_alpha", args.lora_alpha)
        final_args.lora_dropout = params.get("lora_dropout", args.lora_dropout)
        
        lora_cfg = get_lora_config(final_args, base_model)
        base_model = get_peft_model(base_model, lora_cfg)
        base_model.print_trainable_parameters()

    model = build_model(base_model, head_params, final_args).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=params.get("encoder_lr", args.encoder_lr),
        weight_decay=params.get("encoder_weight_decay", args.encoder_weight_decay),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        save_total_limit=None,  # Keep all checkpoints during training
        warmup_steps=100,
        report_to="none",
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        head_params=head_params,
        encoder_params=encoder_params,
    )

    # Train the model
    train_result = trainer.train()
    
    # Use the final model (last epoch)
    trained_model = trainer.model
    print(f"Training completed: {args.num_train_epochs} epochs")

    # ---------------- Saving ----------------------------------------------------
    encoder_dir = os.path.join(args.output_dir, "encoder")
    head_dir = os.path.join(args.output_dir, "head")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    # Save backbone (and LoRA adapter if present)
    if args.use_lora and isinstance(trained_model.encoder, PeftModel):
        print("Saving LoRA adapter + base model …")
        trained_model.encoder.save_pretrained(encoder_dir)
    else:
        print("Saving transformer backbone …")
        trained_model.encoder.save_pretrained(encoder_dir) # モデルの状態辞書からencoder部分のみを抽出

    # Save tokenizer
    print("Saving tokenizer …")
    tokenizer.save_pretrained(encoder_dir)
    
    # Save head weights separately
    print("Saving head weights …")
    # モデルの状態辞書からhead部分のみを抽出
    full_state_dict = trained_model.state_dict()
    head_state_dict = {}
    
    for key, value in full_state_dict.items():
        if key.startswith('head.'):
            # head.プレフィックスを除去
            head_key = key[5:]  # 'head.'の5文字を除去
            head_state_dict[head_key] = value
    
    print(f"Head state dict keys: {list(head_state_dict.keys())}")
    save_file(head_state_dict, os.path.join(head_dir, "head_weights.safetensors"))
    
    # Verify the saved files
    print("Verifying saved files:")
    print(f"  - Encoder dir: {encoder_dir}")
    encoder_files = os.listdir(encoder_dir)
    print(f"    Files: {encoder_files}")
    
    print(f"  - Head dir: {head_dir}")
    head_files = os.listdir(head_dir)
    print(f"    Files: {head_files}")
    
    # Check if head weights were saved correctly
    head_weights_path = os.path.join(head_dir, "head_weights.safetensors")
    if os.path.exists(head_weights_path):
        try:
            from safetensors import safe_open
            with safe_open(head_weights_path, framework="pt", device="cpu") as f:
                saved_keys = list(f.keys())
            print(f"    Head weights keys: {saved_keys}")
        except Exception as e:
            print(f"    Error reading head weights: {e}")
    else:
        print(f"    ERROR: Head weights file not found!")

    # Add training information to params
    training_info = {
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size
    }
    params["training_info"] = training_info
    
    with open(os.path.join(args.output_dir, "hyperparameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Final verification of the output directory structure
    print(f"\n--- Final Output Directory Structure ---")
    print(f"Output directory: {args.output_dir}")
    
    if os.path.exists(args.output_dir):
        for root, dirs, files in os.walk(args.output_dir):
            level = root.replace(args.output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    print(f"\nModel saved to {args.output_dir} (LoRA: {args.use_lora})")


# --- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune a pretrained protein model for regression with optional LoRA.",
    )

    # Data / model paths
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Architecture
    parser.add_argument("--embedding_type", choices=["cls", "mean"], default="mean")
    parser.add_argument("--head_type", choices=["mlp", "ridge"], default="ridge")

    # Training / optimisation
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--optimize", action="store_true", help="Enable Optuna optimization for encoder/LoRA.")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # -------- LoRA options (default: disabled) ----------------------------------
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine‑tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.2, help="LoRA dropout")

    # -------- Head hyperparameters ----------------------------------------------
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate for the regression head")
    parser.add_argument("--head_weight_decay", type=float, default=0.001, help="Weight decay for the regression head")
    parser.add_argument("--head_dropout_rate", type=float, default=0.2, help="Dropout rate for the regression head")
    parser.add_argument("--head_activate_fnc", type=str, default="ReLU", help="Activation function for the regression head")

    # -------- Encoder hyperparameters -------------------------------------------
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help="Learning rate for the encoder")
    parser.add_argument("--encoder_weight_decay", type=float, default=0.01, help="Weight decay for the encoder")

    args = parser.parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data ----------------------------------------------------------------------
    print("--- Loading data …")
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"tm": "labels", "sequence_aho_ungapped": "text"})
        if "text" not in df.columns or "labels" not in df.columns:
            raise KeyError("CSV file must contain 'seq' and 'tm' columns.")
        df["labels"] = pd.to_numeric(df["labels"], errors="coerce")
        df["text"] = df["text"].astype(str).str.strip()
        df.loc[df["text"] == "", "text"] = np.nan
        df.dropna(subset=["labels", "text"], inplace=True)
        return df

    df_train = clean_dataframe(pd.read_csv(args.train_data_path))
    df_val = clean_dataframe(pd.read_csv(args.val_data_path))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(df_train).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(df_val).map(tokenize_fn, batched=True)
    print(f"Dataset sizes – train: {len(train_ds)} | val: {len(val_ds)}")

    columns_to_use = ['input_ids', 'attention_mask', 'labels']
    train_ds.set_format(type='torch', columns=columns_to_use)
    val_ds.set_format(type='torch', columns=columns_to_use)
    print("Dataset format set to 'torch' with required columns only.")

    # --- Hyper‑parameters -----------------------------------------------------------
    # 1. Determine head parameters from command-line arguments
    head_params = {
        "lr": args.head_lr,
        "weight_decay": args.head_weight_decay,
        "dropout_rate": args.head_dropout_rate,
        "activate_fnc": args.head_activate_fnc,
    }
    params = {"head": head_params}

    # 2. Determine encoder parameters from command-line arguments
    if args.optimize:
        print("--- Optimizing encoder (and LoRA) hyperparameters with Optuna ---")
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda t: objective(t, args, train_ds, val_ds, head_params, tokenizer), n_trials=args.n_trials)
        
        best_hyperparams = study.best_trial.params

        # Save Optuna's best params
        params["encoder"] = {
            "lr": best_hyperparams["encoder_lr"],
            "weight_decay": best_hyperparams["encoder_weight_decay"],
        }
        if args.use_lora:
            params["lora_r"] = best_hyperparams["lora_r"]
            params["lora_alpha"] = best_hyperparams["lora_alpha"]
            params["lora_dropout"] = best_hyperparams["lora_dropout"]
        
        optuna_output_file = os.path.join(args.output_dir, "optuna_best_encoder_params.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(optuna_output_file, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)
        print(f"Saved best Optuna params to {optuna_output_file}")

    else:
        # use command-line specified hyperparameters
        print("--- Using command-line specified hyperparameters ---")
        params["encoder"] = {
            "lr": args.encoder_lr,
            "weight_decay": args.encoder_weight_decay,
        }
        print("Head params:", json.dumps(params['head'], indent=2))
        print("Encoder params:", json.dumps(params['encoder'], indent=2))

    # 3. Start final training
    train_final(params, args, train_ds, val_ds, tokenizer)
    print("--- Done ---")


if __name__ == "__main__":
    main()