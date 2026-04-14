#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8
"""
A script to fine-tune a pre-trained model for protein sequence regression tasks.

Overview:
1.  **Data Loading**: By default loads the Hugging Face dataset ZYMScott/thermo-seq
    (train and validation splits). Alternatively, you can provide custom CSV files
    via --train_data_path and --val_data_path; in that case use --label_column and
    --text_column to specify the column names for the target (e.g. thermal stability)
    and the sequence.
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
    -   **Head-only**: By using the `--head_only` flag, freezes the encoder and trains
        only the regression head (no PLM fine-tuning).
4.  **Hyperparameter Optimization (Optional)**:
    -   If the `--optimize` flag is specified, it uses Optuna to search for the
        optimal hyperparameters for the encoder (lr, weight_decay) and LoRA (r, alpha, dropout).
        The head parameters are fixed based on command-line arguments or defaults.
    -   If not optimizing, it uses values from command-line arguments or defaults.
5.  **Training and Evaluation**:
    -   Trains the final model using the determined hyperparameters.
    -   Evaluates performance on the evaluation dataset after each epoch and saves
        the best-performing model.
    -   Labels (e.g. Tm) are scaled with MaxAbsScaler before training; at evaluation
        and inference, predictions are inverse-transformed to the original scale.
6.  **Model Saving**:
    -   Saves the trained model components (transformer backbone/LoRA adapters,
        regression head), the tokenizer, and the hyperparameters used to a specified
        output directory.
    -   If LoRA was used, the adapter is saved relative to the base model, creating a
        self-contained and easily reloadable directory.
    -   If Optuna was used, the best parameters for the encoder are saved to a JSON file.

Command-line Arguments:
    --train_data_path (str, optional):
        Path to the CSV file for training. If omitted (with --val_data_path), data is
        loaded from the Hugging Face dataset (see --dataset_name).
    --val_data_path (str, optional):
        Path to the CSV file for validation. Must be used together with --train_data_path.
    --dataset_name (str):
        Hugging Face dataset name for train/validation when not using CSV. Default: "ZYMScott/thermo-seq"
    --label_column (str):
        CSV column name for the target (e.g. tm / thermal stability). Used only with CSV. Default: "tm"
    --text_column (str):
        CSV column name for the sequence. Used only with CSV. Default: "sequence_aho_ungapped"
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
    --n_gpus (int):
        Number of GPUs for parallel Optuna trials. When > 1, trials are distributed across
        GPUs via multiprocessing with shared SQLite storage. Default: 1
    --max_length (int):
        Maximum sequence length for tokenization. Longer sequences will be truncated. Default: 256
    --seed (int):
        Random seed for reproducibility. Default: 42
    --use_lora (bool):
        If set, enables parameter-efficient fine-tuning using LoRA. Default: False
    --head_only (bool):
        If set, freezes the encoder and trains only the regression head (no PLM fine-tuning).
        Cannot be used together with --use_lora. Default: False
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
import sys
import json
import random
import argparse
import gc
import glob
import shutil
import subprocess
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset, load_dataset
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MaxAbsScaler
import joblib
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
        outputs = model(**inputs)  # labels are included in inputs
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

def make_compute_metrics(scaler):
    """Return a compute_metrics function that inverse-transforms predictions and labels to original scale."""
    def compute_metrics(pred):
        labels = pred.label_ids.reshape(-1)
        preds = pred.predictions.reshape(-1)
        labels_orig = scaler.inverse_transform(np.asarray(labels).reshape(-1, 1)).ravel()
        preds_orig = scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).ravel()
        mse = mean_squared_error(labels_orig, preds_orig)
        return {"mse": mse, "rmse": np.sqrt(mse), "mae": mean_absolute_error(labels_orig, preds_orig), "r2": r2_score(labels_orig, preds_orig)}
    return compute_metrics


def clean_dataframe(df: pd.DataFrame, label_column: str = "tm", text_column: str = "sequence_aho_ungapped") -> pd.DataFrame:
    """Normalize CSV columns to 'text' and 'labels', drop nulls. Used only when loading from CSV."""
    df = df.rename(columns={label_column: "labels", text_column: "text"})
    if "text" not in df.columns or "labels" not in df.columns:
        raise KeyError(f"CSV must contain columns '{label_column}' (or 'labels') and '{text_column}' (or 'text').")
    df["labels"] = pd.to_numeric(df["labels"], errors="coerce")
    df["text"] = df["text"].astype(str).str.strip()
    df.loc[df["text"] == "", "text"] = np.nan
    df.dropna(subset=["labels", "text"], inplace=True)
    return df


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

def objective(trial, args, train_ds, eval_ds, head_params, tokenizer, scaler):
    optuna_params = getattr(args, "optuna_params", None)
    if optuna_params is None:
        if args.head_only:
            optuna_params = ["head_lr", "head_weight_decay", "batch_size"]
        else:
            optuna_params = ["encoder_lr", "encoder_weight_decay"] + (["lora_r", "lora_alpha", "lora_dropout"] if args.use_lora else [])

    encoder_lr_min = getattr(args, "optuna_encoder_lr_min", 1e-5)
    encoder_lr_max = getattr(args, "optuna_encoder_lr_max", 1e-3)
    encoder_wd_min = getattr(args, "optuna_encoder_weight_decay_min", 1e-4)
    encoder_wd_max = getattr(args, "optuna_encoder_weight_decay_max", 1e-1)

    if "encoder_lr" in optuna_params:
        encoder_lr = trial.suggest_float("encoder_lr", encoder_lr_min, encoder_lr_max)
    else:
        encoder_lr = args.encoder_lr
    if "encoder_weight_decay" in optuna_params:
        encoder_weight_decay = trial.suggest_float("encoder_weight_decay", encoder_wd_min, encoder_wd_max)
    else:
        encoder_weight_decay = args.encoder_weight_decay

    encoder_params = {"lr": encoder_lr, "weight_decay": encoder_weight_decay}

    trial_head_params = dict(head_params)
    if "head_lr" in optuna_params:
        h_lr_min = getattr(args, "optuna_head_lr_min", 1e-4)
        h_lr_max = getattr(args, "optuna_head_lr_max", 1e-2)
        trial_head_params["lr"] = trial.suggest_float("head_lr", h_lr_min, h_lr_max)
    if "head_weight_decay" in optuna_params:
        h_wd_min = getattr(args, "optuna_head_weight_decay_min", 0.0)
        h_wd_max = getattr(args, "optuna_head_weight_decay_max", 0.1)
        trial_head_params["weight_decay"] = trial.suggest_float("head_weight_decay", h_wd_min, h_wd_max)

    if "batch_size" in optuna_params:
        batch_size_str = getattr(args, "optuna_batch_size", "8,16,32")
        batch_choices = [int(x.strip()) for x in batch_size_str.split(",") if x.strip()]
        if not batch_choices:
            batch_choices = [8, 16, 32]
        trial_batch_size = trial.suggest_categorical("batch_size", batch_choices)
    else:
        trial_batch_size = args.batch_size

    base_model = AutoModel.from_pretrained(args.model_path)
    trial_args = deepcopy(args)

    if args.head_only:
        for p in base_model.parameters():
            p.requires_grad = False

    if not args.head_only and args.use_lora:
        if get_peft_model is None:
            raise ImportError("peft is not installed but --use_lora was passed.")
        
        lora_r_str = getattr(args, "optuna_lora_r", "4,8,16,32")
        lora_alpha_str = getattr(args, "optuna_lora_alpha", "16,32,64,128")
        lora_r_choices = [int(x.strip()) for x in lora_r_str.split(",") if x.strip()]
        lora_alpha_choices = [int(x.strip()) for x in lora_alpha_str.split(",") if x.strip()]
        lora_dropout_min = getattr(args, "optuna_lora_dropout_min", 0.0)
        lora_dropout_max = getattr(args, "optuna_lora_dropout_max", 0.5)
        if not lora_r_choices:
            lora_r_choices = [4, 8, 16, 32]
        if not lora_alpha_choices:
            lora_alpha_choices = [16, 32, 64, 128]

        if "lora_r" in optuna_params:
            trial_args.lora_r = trial.suggest_categorical("lora_r", lora_r_choices)
        else:
            trial_args.lora_r = args.lora_r
        if "lora_alpha" in optuna_params:
            trial_args.lora_alpha = trial.suggest_categorical("lora_alpha", lora_alpha_choices)
        else:
            trial_args.lora_alpha = args.lora_alpha
        if "lora_dropout" in optuna_params:
            trial_args.lora_dropout = trial.suggest_float("lora_dropout", lora_dropout_min, lora_dropout_max)
        else:
            trial_args.lora_dropout = args.lora_dropout
        
        lora_cfg = get_lora_config(trial_args, base_model)
        base_model = get_peft_model(base_model, lora_cfg)
        base_model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(base_model, trial_head_params, trial_args).to(device)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/trial_{trial.number}",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=trial_batch_size,
        per_device_eval_batch_size=trial_batch_size,
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
        compute_metrics=make_compute_metrics(scaler),
        tokenizer=tokenizer,
        head_params=trial_head_params,
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
    trial_checkpoint_dir = training_args.output_dir
    if os.path.exists(trial_checkpoint_dir):
        shutil.rmtree(trial_checkpoint_dir)
        print(f"[Trial {trial.number}] Removed trial directory: {trial_checkpoint_dir}")

    del model, trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.get("eval_rmse", float("inf"))


# --- Worker for multi-GPU Optuna (subprocess sets CUDA before torch import) ---

OPTUNA_STUDY_NAME = "plm_sft_study"
# Param names that can be passed to --optuna_params (lora_* only valid when --use_lora)
OPTUNA_PARAM_NAMES = ("encoder_lr", "encoder_weight_decay", "head_lr", "head_weight_decay", "batch_size", "lora_r", "lora_alpha", "lora_dropout")


# run_optimization_worker is only used for the doc/flow; actual multi-GPU uses subprocess.
def run_optimization_worker(gpu_id, storage_url, n_trials_this_worker, args, train_ds, eval_ds, head_params, tokenizer, scaler):
    """Run Optuna optimization in a worker process bound to a single GPU (used when invoked as subprocess)."""
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage_url)
    study.optimize(
        lambda t: objective(t, args, train_ds, eval_ds, head_params, tokenizer, scaler),
        n_trials=n_trials_this_worker,
    )


# --- Final training & saving ----------------------------------------------------

N_REEVAL_CONFIGS = 5
N_REEVAL_SEEDS = 3


def get_params_from_trial(trial, args, head_params_base, optuna_params_resolved):
    """Build the full params dict used by train_final from an Optuna trial's params."""
    hp = trial.params
    if args.head_only:
        params = {
            "encoder": {"lr": args.encoder_lr, "weight_decay": args.encoder_weight_decay},
            "head": dict(head_params_base),
        }
    else:
        params = {
            "encoder": {
                "lr": hp.get("encoder_lr", args.encoder_lr),
                "weight_decay": hp.get("encoder_weight_decay", args.encoder_weight_decay),
            },
            "head": dict(head_params_base),
        }
    if "head_lr" in hp:
        params["head"]["lr"] = hp["head_lr"]
    if "head_weight_decay" in hp:
        params["head"]["weight_decay"] = hp["head_weight_decay"]
    if "batch_size" in optuna_params_resolved and "batch_size" in hp:
        params["batch_size"] = hp["batch_size"]
    if not args.head_only and args.use_lora:
        params["lora_r"] = hp.get("lora_r", args.lora_r)
        params["lora_alpha"] = hp.get("lora_alpha", args.lora_alpha)
        params["lora_dropout"] = hp.get("lora_dropout", args.lora_dropout)
    return params


def _ensure_single_process_env():
    """Set env vars so TrainingArguments/Accelerate treat this process as single-process (no distributed)."""
    os.environ["WORLD_SIZE"] = "1"
    for key in ("RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
        os.environ.pop(key, None)


def run_reeval_worker(args, train_ds, val_ds, head_params, tokenizer, scaler):
    """Run a subset of reeval (config, seed) tasks and write results to a JSON file.
    Used when this process is launched as a subprocess with --reeval_worker.
    """
    _ensure_single_process_env()
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=args.reeval_storage)
    with open(args.reeval_tasks_file, "r") as f:
        tasks = json.load(f)  # list of {"config_i": int, "seed": int}

    if args.head_only:
        optuna_params_resolved = args.optuna_params if args.optuna_params is not None else ["head_lr", "head_weight_decay", "batch_size"]
    else:
        optuna_params_resolved = args.optuna_params if args.optuna_params is not None else (
            ["encoder_lr", "encoder_weight_decay"] + (["lora_r", "lora_alpha", "lora_dropout"] if args.use_lora else [])
        )
    completed = [t for t in study.get_trials() if t.value is not None]
    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials = sorted_trials[:N_REEVAL_CONFIGS]

    results = []  # list of {"config_i": i, "seed": s, "rmse": float, "save_subdir": str}
    for task in tasks:
        config_i = task["config_i"]
        seed = task["seed"]
        trial = top_trials[config_i]
        params_i = get_params_from_trial(trial, args, head_params, optuna_params_resolved)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        save_subdir = os.path.join(args.output_dir, "reeval", f"c{config_i}_s{seed}")
        os.makedirs(save_subdir, exist_ok=True)
        rmse = train_final(
            params_i, args, train_ds, val_ds, tokenizer, scaler,
            output_dir_override=save_subdir,
            save_total_limit_override=1,
        )
        results.append({"config_i": config_i, "seed": seed, "rmse": float(rmse), "save_subdir": save_subdir})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reeval_dir = os.path.join(args.output_dir, "reeval")
    os.makedirs(reeval_dir, exist_ok=True)
    out_path = os.path.join(reeval_dir, f"worker_{args.reeval_worker_id}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def run_reeval_from_study(study, args, train_ds, val_ds, head_params, tokenizer, scaler, n_gpus_reeval=1, storage_url=None):
    """Re-evaluate top trials with multiple seeds and save best run.

    This does NOT execute new Optuna trials; it consumes an existing Study instance
    (e.g., loaded from optuna_study.db) and performs the 'reeval' stage only.
    When n_gpus_reeval > 1 and storage_url is set, runs (config, seed) tasks in parallel
    across multiple GPU worker processes.
    """
    # Ensure this process (and any subprocess we spawn) does not trigger distributed/multinode
    # when creating TrainingArguments (e.g. after multi-GPU Optuna left WORLD_SIZE set).
    if n_gpus_reeval <= 1:
        _ensure_single_process_env()
    if args.head_only:
        optuna_params_resolved = args.optuna_params if args.optuna_params is not None else ["head_lr", "head_weight_decay", "batch_size"]
    else:
        optuna_params_resolved = args.optuna_params if args.optuna_params is not None else (
            ["encoder_lr", "encoder_weight_decay"] + (["lora_r", "lora_alpha", "lora_dropout"] if args.use_lora else [])
        )

    completed = [t for t in study.get_trials() if t.value is not None]
    if not completed:
        raise ValueError("No completed Optuna trials to re-evaluate.")
    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials = sorted_trials[:N_REEVAL_CONFIGS]
    seeds = [args.seed + k for k in range(N_REEVAL_SEEDS)]

    n_reeval_runs = len(top_trials) * len(seeds)
    print(f"[Reeval] Re-training top {len(top_trials)} configs with {len(seeds)} seeds each ({n_reeval_runs} runs total). This can take a long time.")

    results = {}
    if n_gpus_reeval > 1 and storage_url is not None:
        # Parallel reeval: split (config_i, seed) tasks across worker processes
        tasks = [{"config_i": i, "seed": s} for i in range(len(top_trials)) for s in seeds]
        n_workers = min(n_gpus_reeval, len(tasks))
        if n_workers <= 0:
            n_workers = 1
        reeval_dir = os.path.join(args.output_dir, "reeval")
        os.makedirs(reeval_dir, exist_ok=True)
        # Chunk tasks for each worker
        chunk_size = (len(tasks) + n_workers - 1) // n_workers
        task_chunks = [tasks[k * chunk_size:(k + 1) * chunk_size] for k in range(n_workers)]
        # Drop empty chunks (if tasks < n_workers)
        task_chunks = [c for c in task_chunks if c]
        n_workers = len(task_chunks)
        print(f"[Reeval] Using {n_workers} GPU workers for parallel re-evaluation.")
        task_files = []
        for k in range(n_workers):
            tf = os.path.join(reeval_dir, f"reeval_tasks_{k}.json")
            with open(tf, "w") as f:
                json.dump(task_chunks[k], f, indent=2)
            task_files.append(tf)
        procs = []
        for k in range(n_workers):
            # Workers must see single-process env so TrainingArguments does not require MASTER_ADDR
            env = {k_: v for k_, v in os.environ.items() if k_ not in ("RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "WORLD_SIZE")}
            env["CUDA_VISIBLE_DEVICES"] = str(k)
            env["WORLD_SIZE"] = "1"
            cmd = [sys.executable, "-u", __file__] + sys.argv[1:] + [
                "--reeval_worker",
                "--reeval_worker_id", str(k),
                "--reeval_tasks_file", task_files[k],
                "--reeval_storage", storage_url,
            ]
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)
        for p in procs:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Reeval worker process exited with code {p.returncode}")
        # Merge worker results into results[i][seed] = (rmse, save_subdir)
        for i in range(len(top_trials)):
            results[i] = {}
        for k in range(n_workers):
            res_path = os.path.join(reeval_dir, f"worker_{k}_results.json")
            with open(res_path, "r") as f:
                worker_results = json.load(f)
            for r in worker_results:
                config_i = r["config_i"]
                seed = r["seed"]
                results[config_i][seed] = (r["rmse"], r["save_subdir"])
    else:
        # Sequential reeval (single process)
        run_idx = 0
        for i, trial in enumerate(top_trials):
            params_i = get_params_from_trial(trial, args, head_params, optuna_params_resolved)
            results[i] = {}
            for seed in seeds:
                run_idx += 1
                print(f"[Reeval] Run {run_idx}/{n_reeval_runs}: config {i+1}/{len(top_trials)}, seed {seed}")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                save_subdir = os.path.join(args.output_dir, "reeval", f"c{i}_s{seed}")
                os.makedirs(save_subdir, exist_ok=True)
                rmse = train_final(
                    params_i, args, train_ds, val_ds, tokenizer, scaler,
                    output_dir_override=save_subdir,
                    save_total_limit_override=1,
                )
                results[i][seed] = (rmse, save_subdir)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Reeval] Run {run_idx}/{n_reeval_runs} done: RMSE={rmse:.4f}")

    mean_rmse = [np.mean([results[i][s][0] for s in seeds]) for i in range(len(top_trials))]
    best_config_idx = int(np.argmin(mean_rmse))
    best_seed = min(seeds, key=lambda s: results[best_config_idx][s][0])
    best_subdir = results[best_config_idx][best_seed][1]

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("encoder", "head", "checkpoints"):
        dst = os.path.join(args.output_dir, sub)
        src = os.path.join(best_subdir, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.exists(src):
            shutil.copytree(src, dst)
    for fname in ("label_scaler.joblib", "hyperparameters.json"):
        shutil.copy2(os.path.join(best_subdir, fname), os.path.join(args.output_dir, fname))

    with open(os.path.join(args.output_dir, "optuna_best_encoder_params.json"), "w") as f:
        json.dump(top_trials[best_config_idx].params, f, indent=4)
    best_trial_info = {
        "config_index": best_config_idx,
        "chosen_seed": best_seed,
        "mean_rmse": float(mean_rmse[best_config_idx]),
        "seed_scores": [results[best_config_idx][s][0] for s in seeds],
        "all_config_mean_rmse": [float(m) for m in mean_rmse],
    }
    with open(os.path.join(args.output_dir, "best_trial_info.json"), "w") as f:
        json.dump(best_trial_info, f, indent=2)

    reeval_dir = os.path.join(args.output_dir, "reeval")
    if os.path.exists(reeval_dir):
        shutil.rmtree(reeval_dir)
    print(f"Re-evaluation done. Best config index {best_config_idx}, seed {best_seed}, mean_rmse={mean_rmse[best_config_idx]:.4f}")


def train_final(params, args, train_ds, eval_ds, tokenizer, scaler, output_dir_override=None, save_total_limit_override=None):
    print("--- Starting final model training ---")
    save_dir = output_dir_override if output_dir_override is not None else args.output_dir
    encoder_params = params.get("encoder", {})
    head_params = params.get("head", {})
    batch_size = params.get("batch_size", args.batch_size)
    print("Encoder params:", json.dumps(encoder_params, indent=2))
    print("Head params:", json.dumps(head_params, indent=2))
    print("Batch size:", batch_size)

    base_model = AutoModel.from_pretrained(args.model_path)
    final_args = deepcopy(args)

    if args.head_only:
        for p in base_model.parameters():
            p.requires_grad = False

    if args.use_lora:
        # If LoRA params were optimized or loaded, use them
        final_args.lora_r = params.get("lora_r", args.lora_r)
        final_args.lora_alpha = params.get("lora_alpha", args.lora_alpha)
        final_args.lora_dropout = params.get("lora_dropout", args.lora_dropout)
        
        lora_cfg = get_lora_config(final_args, base_model)
        base_model = get_peft_model(base_model, lora_cfg)
        base_model.print_trainable_parameters()

    model = build_model(base_model, head_params, final_args).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=params.get("encoder_lr", args.encoder_lr),
        weight_decay=params.get("encoder_weight_decay", args.encoder_weight_decay),
        load_best_model_at_end=True,
        metric_for_best_model="eval_rmse",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        save_total_limit=save_total_limit_override if save_total_limit_override is not None else None,
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
        compute_metrics=make_compute_metrics(scaler),
        tokenizer=tokenizer,
        head_params=head_params,
        encoder_params=encoder_params,
    )

    # Train the model
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    
    # Use the final model (last epoch)
    trained_model = trainer.model
    print(f"Training completed: {args.num_train_epochs} epochs")

    # ---------------- Saving ----------------------------------------------------
    encoder_dir = os.path.join(save_dir, "encoder")
    head_dir = os.path.join(save_dir, "head")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    # Save label scaler for inference (inverse_transform)
    scaler_path = os.path.join(save_dir, "label_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saving label scaler to {scaler_path} …")

    # Save backbone (and LoRA adapter if present)
    if args.use_lora and isinstance(trained_model.encoder, PeftModel):
        print("Saving LoRA adapter + base model …")
        trained_model.encoder.save_pretrained(encoder_dir)
    else:
        print("Saving transformer backbone …")
        trained_model.encoder.save_pretrained(encoder_dir)

    # Save tokenizer
    print("Saving tokenizer …")
    tokenizer.save_pretrained(encoder_dir)
    
    # Save head weights separately
    print("Saving head weights …")
    # Extract only the head submodule from the full state dict
    full_state_dict = trained_model.state_dict()
    head_state_dict = {}

    for key, value in full_state_dict.items():
        if key.startswith('head.'):
            head_key = key[5:]  # Remove 'head.' prefix
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
        "batch_size": params.get("batch_size", args.batch_size)
    }
    params["training_info"] = training_info
    params["label_scaler_scale"] = scaler.scale_.tolist()
    
    with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Final verification of the output directory structure
    print(f"\n--- Final Output Directory Structure ---")
    print(f"Output directory: {save_dir}")
    
    if os.path.exists(save_dir):
        for root, dirs, files in os.walk(save_dir):
            level = root.replace(save_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    print(f"\nModel saved to {save_dir} (Head-only: {args.head_only}, LoRA: {args.use_lora})")
    return eval_metrics.get("eval_rmse", float("inf"))


# --- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune a pretrained protein model for regression with optional LoRA.",
    )

    # Data / model paths
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to training CSV. Omit to use Hugging Face dataset.")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation CSV. Must set with --train_data_path when using CSV.")
    parser.add_argument("--dataset_name", type=str, default="ZYMScott/thermo-seq", help="Hugging Face dataset name when not using CSV.")
    parser.add_argument("--label_column", type=str, default="tm", help="CSV column name for target (e.g. tm). Used only with CSV.")
    parser.add_argument("--text_column", type=str, default="sequence_aho_ungapped", help="CSV column name for sequence. Used only with CSV.")
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
    parser.add_argument("--reeval_only", action="store_true",
        help="Skip Optuna optimization and run only the re-evaluation stage from an existing optuna_study.db.")
    parser.add_argument("--optuna_study_path", type=str, default=None,
        help="Path to optuna_study.db (SQLite). If omitted, uses <output_dir>/optuna_study.db when --reeval_only is set.")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs for parallel Optuna trials (only when --optimize). Default: 1")
    parser.add_argument("--reeval_n_gpus", type=int, default=1,
        help="Number of GPUs for parallel re-evaluation (top configs × seeds). Used with --optimize (after Optuna) or --reeval_only. Default: 1")
    # Optuna search space (used only when --optimize)
    parser.add_argument("--optuna_encoder_lr_min", type=float, default=1e-5, help="Optuna: encoder LR lower bound (default: 1e-5)")
    parser.add_argument("--optuna_encoder_lr_max", type=float, default=1e-3, help="Optuna: encoder LR upper bound (default: 1e-3)")
    parser.add_argument("--optuna_encoder_weight_decay_min", type=float, default=1e-4, help="Optuna: encoder weight_decay lower bound (default: 1e-4)")
    parser.add_argument("--optuna_encoder_weight_decay_max", type=float, default=1e-1, help="Optuna: encoder weight_decay upper bound (default: 1e-1)")
    parser.add_argument("--optuna_lora_r", type=str, default="4,8,16,32", help="Optuna: LoRA r candidates, comma-separated (default: 4,8,16,32). Used only with --use_lora.")
    parser.add_argument("--optuna_lora_alpha", type=str, default="16,32,64,128", help="Optuna: LoRA alpha candidates, comma-separated (default: 16,32,64,128). Used only with --use_lora.")
    parser.add_argument("--optuna_lora_dropout_min", type=float, default=0.0, help="Optuna: LoRA dropout lower bound (default: 0.0). Used only with --use_lora.")
    parser.add_argument("--optuna_lora_dropout_max", type=float, default=0.5, help="Optuna: LoRA dropout upper bound (default: 0.5). Used only with --use_lora.")
    parser.add_argument("--optuna_head_lr_min", type=float, default=1e-4, help="Optuna: head LR lower bound (default: 1e-4). Used when head_lr is in --optuna_params.")
    parser.add_argument("--optuna_head_lr_max", type=float, default=1e-2, help="Optuna: head LR upper bound (default: 1e-2). Used when head_lr is in --optuna_params.")
    parser.add_argument("--optuna_head_weight_decay_min", type=float, default=0.0, help="Optuna: head weight_decay lower bound (default: 0.0). Used when head_weight_decay is in --optuna_params.")
    parser.add_argument("--optuna_head_weight_decay_max", type=float, default=0.1, help="Optuna: head weight_decay upper bound (default: 0.1). Used when head_weight_decay is in --optuna_params.")
    parser.add_argument("--optuna_batch_size", type=str, default="8,16,32", help="Optuna: batch_size candidates, comma-separated (default: 8,16,32). Used when batch_size is in --optuna_params.")
    parser.add_argument("--optuna_params", nargs="*", default=None, metavar="PARAM",
        help="Which parameters to optimize. Choices: encoder_lr, encoder_weight_decay, head_lr, head_weight_decay, batch_size, lora_r, lora_alpha, lora_dropout (lora_* require --use_lora). Default: encoder_lr encoder_weight_decay, plus lora_r lora_alpha lora_dropout when --use_lora.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # Internal: used when this script is run as an Optuna worker subprocess (do not set manually)
    parser.add_argument("--optuna_storage", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--optuna_n_trials_this_worker", type=int, default=None, help=argparse.SUPPRESS)
    # Internal: used when this script is run as a reeval worker subprocess (do not set manually)
    parser.add_argument("--reeval_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reeval_worker_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reeval_tasks_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reeval_storage", type=str, default=None, help=argparse.SUPPRESS)

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
    parser.add_argument("--head_only", action="store_true",
        help="Only train the regression head; freeze the encoder (no PLM fine-tuning).")
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help="Learning rate for the encoder")
    parser.add_argument("--encoder_weight_decay", type=float, default=0.01, help="Weight decay for the encoder")

    args = parser.parse_args()

    if args.head_only and args.use_lora:
        raise ValueError("--head_only and --use_lora cannot be used together.")
    if args.optimize and args.reeval_only:
        raise ValueError("--optimize and --reeval_only cannot be used together.")

    # Require both or neither of train_data_path and val_data_path
    if args.train_data_path is not None and args.val_data_path is None:
        raise ValueError("If --train_data_path is set, --val_data_path must also be set.")
    if args.train_data_path is None and args.val_data_path is not None:
        raise ValueError("If --val_data_path is set, --train_data_path must also be set.")

    if args.optimize:
        if args.optuna_encoder_lr_min > args.optuna_encoder_lr_max:
            raise ValueError("optuna_encoder_lr_min must be <= optuna_encoder_lr_max")
        if args.optuna_encoder_weight_decay_min > args.optuna_encoder_weight_decay_max:
            raise ValueError("optuna_encoder_weight_decay_min must be <= optuna_encoder_weight_decay_max")
        if args.head_only:
            optuna_params = args.optuna_params if args.optuna_params is not None else ["head_lr", "head_weight_decay", "batch_size"]
            for p in optuna_params:
                if p not in ("head_lr", "head_weight_decay", "batch_size"):
                    raise ValueError(f"--head_only only allows optimizing head_lr, head_weight_decay, batch_size; got {p}")
        else:
            optuna_params = args.optuna_params if args.optuna_params is not None else (
                ["encoder_lr", "encoder_weight_decay"] + (["lora_r", "lora_alpha", "lora_dropout"] if args.use_lora else [])
            )
        for p in optuna_params:
            if p not in OPTUNA_PARAM_NAMES:
                raise ValueError(f"Invalid --optuna_params value: {p}. Choices: {list(OPTUNA_PARAM_NAMES)}")
            if p in ("lora_r", "lora_alpha", "lora_dropout") and not args.use_lora:
                raise ValueError(f"--optuna_params {p} requires --use_lora")
        if "head_lr" in optuna_params and args.optuna_head_lr_min > args.optuna_head_lr_max:
            raise ValueError("optuna_head_lr_min must be <= optuna_head_lr_max when head_lr is in --optuna_params")
        if "head_weight_decay" in optuna_params and args.optuna_head_weight_decay_min > args.optuna_head_weight_decay_max:
            raise ValueError("optuna_head_weight_decay_min must be <= optuna_head_weight_decay_max when head_weight_decay is in --optuna_params")
        if "batch_size" in optuna_params:
            batch_vals = [int(x.strip()) for x in args.optuna_batch_size.split(",") if x.strip()]
            if not batch_vals:
                raise ValueError("optuna_batch_size must contain at least one integer (e.g. 8,16,32) when batch_size is in --optuna_params")
        if args.use_lora:
            lora_r_vals = [int(x.strip()) for x in args.optuna_lora_r.split(",") if x.strip()]
            lora_alpha_vals = [int(x.strip()) for x in args.optuna_lora_alpha.split(",") if x.strip()]
            if not lora_r_vals:
                raise ValueError("optuna_lora_r must contain at least one integer (e.g. 4,8,16,32)")
            if not lora_alpha_vals:
                raise ValueError("optuna_lora_alpha must contain at least one integer (e.g. 16,32,64,128)")
            if args.optuna_lora_dropout_min > args.optuna_lora_dropout_max:
                raise ValueError("optuna_lora_dropout_min must be <= optuna_lora_dropout_max")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data ----------------------------------------------------------------------
    print("--- Loading data …")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    if args.train_data_path is None and args.val_data_path is None:
        # Load from Hugging Face dataset (default)
        ds = load_dataset(args.dataset_name)
        train_ds_raw = ds["train"]
        val_ds_raw = ds["validation"]
        # Map columns: seq -> text, label -> labels (thermo-seq uses 'seq' and 'label')
        train_ds_raw = train_ds_raw.rename_column("seq", "text").rename_column("label", "labels")
        val_ds_raw = val_ds_raw.rename_column("seq", "text").rename_column("label", "labels")
        # Drop rows with null labels or text if any
        train_ds_raw = train_ds_raw.filter(lambda x: x["text"] is not None and x["labels"] is not None and str(x["text"]).strip() != "")
        val_ds_raw = val_ds_raw.filter(lambda x: x["text"] is not None and x["labels"] is not None and str(x["text"]).strip() != "")
        # Scale labels with MaxAbsScaler (fit on train only)
        train_labels = np.array(train_ds_raw["labels"]).reshape(-1, 1)
        scaler = MaxAbsScaler().fit(train_labels)
        train_ds_raw = train_ds_raw.remove_columns("labels").add_column(
            "labels", scaler.transform(train_labels).ravel().tolist()
        )
        val_labels = np.array(val_ds_raw["labels"]).reshape(-1, 1)
        val_ds_raw = val_ds_raw.remove_columns("labels").add_column(
            "labels", scaler.transform(val_labels).ravel().tolist()
        )
        train_ds = train_ds_raw.map(tokenize_fn, batched=True)
        val_ds = val_ds_raw.map(tokenize_fn, batched=True)
        print(f"Loaded Hugging Face dataset '{args.dataset_name}' – train: {len(train_ds)} | val: {len(val_ds)}")
    else:
        # Load from CSV with user-specified column names
        df_train = clean_dataframe(pd.read_csv(args.train_data_path), label_column=args.label_column, text_column=args.text_column)
        df_val = clean_dataframe(pd.read_csv(args.val_data_path), label_column=args.label_column, text_column=args.text_column)
        # Scale labels with MaxAbsScaler (fit on train only)
        train_labels = df_train["labels"].values.reshape(-1, 1)
        scaler = MaxAbsScaler().fit(train_labels)
        df_train["labels"] = scaler.transform(train_labels).ravel()
        df_val["labels"] = scaler.transform(df_val["labels"].values.reshape(-1, 1)).ravel()
        train_ds = Dataset.from_pandas(df_train).map(tokenize_fn, batched=True)
        val_ds = Dataset.from_pandas(df_val).map(tokenize_fn, batched=True)
        print(f"Dataset sizes – train: {len(train_ds)} | val: {len(val_ds)}")

    columns_to_use = ['input_ids', 'attention_mask', 'labels']
    train_ds.set_format(type='torch', columns=columns_to_use)
    val_ds.set_format(type='torch', columns=columns_to_use)
    print("Dataset format set to 'torch' with required columns only.")

    # --- Hyperparameters -----------------------------------------------------------
    # 1. Head parameters from command-line
    head_params = {
        "lr": args.head_lr,
        "weight_decay": args.head_weight_decay,
        "dropout_rate": args.head_dropout_rate,
        "activate_fnc": args.head_activate_fnc,
    }
    params = {"head": head_params}

    # Worker mode: run only Optuna trials then exit (used when launched as subprocess for multi-GPU)
    if getattr(args, "optuna_storage", None) is not None and getattr(args, "optuna_n_trials_this_worker", None) is not None:
        print("--- Optuna worker mode: running trials then exiting ---")
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=args.optuna_storage)
        study.optimize(
            lambda t: objective(t, args, train_ds, val_ds, head_params, tokenizer, scaler),
            n_trials=args.optuna_n_trials_this_worker,
        )
        sys.exit(0)

    # Worker mode: run a subset of reeval (config, seed) tasks then exit (used when launched as subprocess for multi-GPU reeval)
    if getattr(args, "reeval_worker", False) and getattr(args, "reeval_tasks_file", None) is not None and getattr(args, "reeval_storage", None) is not None:
        print("--- Reeval worker mode: running assigned tasks then exiting ---")
        run_reeval_worker(args, train_ds, val_ds, head_params, tokenizer, scaler)
        sys.exit(0)

    # Re-evaluation only mode: load an existing Optuna study and run reeval without new trials
    if args.reeval_only:
        os.makedirs(args.output_dir, exist_ok=True)
        study_path = args.optuna_study_path if args.optuna_study_path is not None else os.path.join(args.output_dir, "optuna_study.db")
        study_path = os.path.abspath(study_path)
        if not os.path.exists(study_path):
            raise FileNotFoundError(
                f"optuna_study.db not found at {study_path}. "
                "Provide --optuna_study_path or ensure <output_dir>/optuna_study.db exists."
            )
        storage_url = f"sqlite:///{study_path}"
        print(f"--- Re-eval only mode: loading Optuna study from {study_path} ---")
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage_url)
        n_gpus_reeval = min(args.reeval_n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if n_gpus_reeval <= 0:
            n_gpus_reeval = 1
        run_reeval_from_study(study, args, train_ds, val_ds, head_params, tokenizer, scaler, n_gpus_reeval=n_gpus_reeval, storage_url=storage_url)
        print("--- Done ---")
        return

    # 2. Encoder parameters from command-line or Optuna
    if args.optimize:
        if args.head_only:
            print("--- Optimizing head hyperparameters with Optuna ---")
        else:
            print("--- Optimizing encoder (and LoRA) hyperparameters with Optuna ---")
        n_gpus = min(args.n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if n_gpus <= 0:
            n_gpus = 1

        if n_gpus > 1:
            # Multi-GPU: shared storage + one subprocess per GPU (env set before import)
            os.makedirs(args.output_dir, exist_ok=True)
            storage_path = os.path.abspath(os.path.join(args.output_dir, "optuna_study.db"))
            storage_url = f"sqlite:///{storage_path}"
            sampler = optuna.samplers.TPESampler(seed=args.seed)
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(),
                storage=storage_url,
                study_name=OPTUNA_STUDY_NAME,
                load_if_exists=False,
            )
            n_trials_base = args.n_trials // n_gpus
            remainder = args.n_trials % n_gpus
            procs = []
            for i in range(n_gpus):
                n_trials_i = n_trials_base + (1 if i < remainder else 0)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
                cmd = [sys.executable, "-u", __file__] + sys.argv[1:] + [
                    "--optuna_storage", storage_url,
                    "--optuna_n_trials_this_worker", str(n_trials_i),
                ]
                p = subprocess.Popen(cmd, env=env)
                procs.append(p)
            for p in procs:
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(f"Optuna worker process exited with code {p.returncode}")
            study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage_url)
            best_hyperparams = study.best_trial.params
            print(f"Multi-GPU Optuna finished ({n_gpus} workers, {args.n_trials} trials total).")
        else:
            # Single-GPU: current in-process optimization
            sampler = optuna.samplers.TPESampler(seed=args.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda t: objective(t, args, train_ds, val_ds, head_params, tokenizer, scaler), n_trials=args.n_trials)
            best_hyperparams = study.best_trial.params

        # Re-evaluate top trials with multiple seeds, then save best run
        n_gpus_reeval = min(args.reeval_n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if n_gpus_reeval <= 0:
            n_gpus_reeval = 1
        storage_url_reeval = None
        if n_gpus_reeval > 1:
            storage_path_reeval = os.path.abspath(os.path.join(args.output_dir, "optuna_study.db"))
            storage_url_reeval = f"sqlite:///{storage_path_reeval}"
        run_reeval_from_study(study, args, train_ds, val_ds, head_params, tokenizer, scaler, n_gpus_reeval=n_gpus_reeval, storage_url=storage_url_reeval)

    else:
        # use command-line specified hyperparameters
        print("--- Using command-line specified hyperparameters ---")
        params["encoder"] = {
            "lr": args.encoder_lr,
            "weight_decay": args.encoder_weight_decay,
        }
        print("Head params:", json.dumps(params['head'], indent=2))
        print("Encoder params:", json.dumps(params['encoder'], indent=2))

        # 3. Final training
        train_final(params, args, train_ds, val_ds, tokenizer, scaler)
    print("--- Done ---")


if __name__ == "__main__":
    main()