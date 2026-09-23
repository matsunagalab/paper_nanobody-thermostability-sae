#!/usr/bin/env python3
"""
テストセット性能のブートストラップ95%信頼区間 (Fig. 3b)

テスト147配列を復元抽出で1000回リサンプリングし、各指標の95%CIをpercentile法で算出。
対象: Pre-trained / SFT の 8M, 35M, 150M, 650M (各モデルのhead予測値を使用)。
Spearman相関の点推定+95%CIを、NbBenchベースライン（esm2 / AntiBERTa2、文献値）と
並べた棒グラフ（Fig. 3b）として可視化する。

生成物（論文に採用された図表のみ）:
    figure/Fig3b_spearman_by_model.png -- Spearman相関の95%CI付きモデル比較棒グラフ
    tables/bootstrap_ci.csv            -- 全指標（RMSE/MAE/R²/Pearson/Spearman）の
                                           点推定+95%CI（Pre-trained/SFT × 4サイズ）

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_bootstrap_ci.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).parent.resolve()
REPO_ROOT   = SCRIPT_DIR / '..'
SFT_DIR     = REPO_ROOT / 'supervised_finetuning/models'
TEST_CSV    = REPO_ROOT / 'data/nbbench/thermo-seq/test.csv'
OUTPUT_DIR  = SCRIPT_DIR / 'outputs/bootstrap_ci'

# NbBench文献値（点推定のみ、CIなし; Spearman相関）
NBBENCH_BASELINES = {'NbBench_esm2': 0.389, 'NbBench_AntiBERTa2': 0.587}

RANDOM_STATE   = 42
N_BOOTSTRAP    = 1000
MODEL_SIZES    = ['8m', '35m', '150m', '650m']
ESM_BASE_NAMES = {
    '8m':   'facebook/esm2_t6_8M_UR50D',
    '35m':  'facebook/esm2_t12_35M_UR50D',
    '150m': 'facebook/esm2_t30_150M_UR50D',
    '650m': 'facebook/esm2_t33_650M_UR50D',
}


# ── モデル関連ユーティリティ ──────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.linear = nn.Linear(w.shape[1], 1)
        with torch.no_grad():
            self.linear.weight.data = w
            self.linear.bias.data   = b
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def get_mean_pooled(model, tokenizer, seqs, device, batch_size=16):
    """mean-pool（CLS・EOS除外）した埋め込みを返す。shape: (N, d_model)"""
    model.eval()
    all_reps = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out = model(**inputs)
        last = out.last_hidden_state          # (B, L, D)
        mask = inputs['attention_mask'].float()
        mask[:, 0] = 0                        # CLS
        for j in range(len(batch)):
            eos = int(mask[j].sum()) - 1
            mask[j, eos] = 0                  # EOS
        emb = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        all_reps.append(emb.cpu())
    return torch.cat(all_reps, dim=0)


def predict_tm(model_dir: Path, emb_test: torch.Tensor) -> np.ndarray:
    """headとscalerを使ってTmを予測する。"""
    hw = load_file(model_dir / 'head/head_weights.safetensors')
    scaler = joblib.load(model_dir / 'label_scaler.joblib')
    head = LinearHead(hw['weight'], hw['bias'])
    head.eval()
    with torch.no_grad():
        y_scaled = head(emb_test.float()).numpy()
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ── 評価指標 ─────────────────────────────────────────────────────────────────

METRICS = ['rmse', 'mae', 'r2', 'pearson', 'spearman']

def compute_metrics(y_true, y_pred):
    return {
        'rmse':     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':      float(mean_absolute_error(y_true, y_pred)),
        'r2':       float(r2_score(y_true, y_pred)),
        'pearson':  float(pearsonr(y_true, y_pred)[0]),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
    }


def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """percentile法 95%CI。返り値: {metric: (point, lower, upper)}"""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    boot = {m: [] for m in METRICS}
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        m = compute_metrics(y_true[idx], y_pred[idx])
        for k in METRICS:
            boot[k].append(m[k])
    point = compute_metrics(y_true, y_pred)
    result = {}
    for k in METRICS:
        arr = np.array(boot[k])
        result[k] = (point[k], float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return result


# ── 可視化（Fig. 3b）───────────────────────────────────────────────────────────

def plot_fig3b(df_ci, out_path):
    """Spearman相関の95%CI付きモデル比較棒グラフ（本文 Fig. 3b）。
    NbBench文献ベースライン（CIなし）+ Pre-trained/SFT各サイズ（bootstrap 95%CI）。"""
    def get_spearman_ci(model_type, size):
        row = df_ci[(df_ci['model_type'] == model_type) & (df_ci['model_size'] == size)].iloc[0]
        return float(row['spearman']), float(row['spearman_lo']), float(row['spearman_hi'])

    models_nbbench   = list(NBBENCH_BASELINES.keys())
    spearman_nbbench = list(NBBENCH_BASELINES.values())

    models_pretrained = [f'Pre-trained_{s.upper()}' for s in MODEL_SIZES]
    models_sft        = [f'SFT_{s.upper()}' for s in MODEL_SIZES]

    spearman_pretrained, err_pretrained_lo, err_pretrained_hi = [], [], []
    spearman_sft,        err_sft_lo,        err_sft_hi        = [], [], []
    for s in MODEL_SIZES:
        pt, lo, hi = get_spearman_ci('Pre-trained', s)
        spearman_pretrained.append(pt); err_pretrained_lo.append(pt - lo); err_pretrained_hi.append(hi - pt)
        pt, lo, hi = get_spearman_ci('SFT', s)
        spearman_sft.append(pt); err_sft_lo.append(pt - lo); err_sft_hi.append(hi - pt)

    x_nbbench    = np.arange(len(models_nbbench))
    x_pretrained = np.arange(len(models_pretrained)) + len(models_nbbench) + 0.5
    x_sft        = np.arange(len(models_sft)) + len(models_nbbench) + len(models_pretrained) + 1.0

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ecolor, ecapsize, ecapthick, elw = '#444444', 2, 0.5, 0.6

    bars1 = ax.bar(x_nbbench, spearman_nbbench, color=['#FEE8C8', '#FDBB84'])
    bars2 = ax.bar(x_pretrained, spearman_pretrained,
                   color=['#CFE3C7', '#8AB884', '#5A9A60', '#457B4F'],
                   yerr=[err_pretrained_lo, err_pretrained_hi],
                   error_kw=dict(ecolor=ecolor, capsize=ecapsize, capthick=ecapthick, elinewidth=elw))
    bars3 = ax.bar(x_sft, spearman_sft,
                   color=['#9ECAE1', '#6BAED6', '#4292C6', '#2171B5'],
                   yerr=[err_sft_lo, err_sft_hi],
                   error_kw=dict(ecolor=ecolor, capsize=ecapsize, capthick=ecapthick, elinewidth=elw))

    for bar, val in zip(bars1, spearman_nbbench):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=4.5)
    for bar, val, hi in zip(bars2, spearman_pretrained, err_pretrained_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, val + hi + 0.010, f'{val:.3f}',
                ha='center', va='bottom', fontsize=4.5)
    for bar, val, hi in zip(bars3, spearman_sft, err_sft_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, val + hi + 0.010, f'{val:.3f}',
                ha='center', va='bottom', fontsize=4.5)

    ax.set_ylabel('Spearman', fontsize=6)
    all_vals = spearman_nbbench + spearman_pretrained + spearman_sft
    all_hi   = [0, 0] + err_pretrained_hi + err_sft_hi
    ymax = max(v + e for v, e in zip(all_vals, all_hi)) * 1.12
    ax.set_ylim(0.35, ymax)
    ax.set_yticks(np.arange(0.35, np.ceil(ymax * 10) / 10 + 0.05, 0.05))
    ax.tick_params(axis='both', direction='out', bottom=True, left=True,
                   length=3, width=0.3, labelsize=5)
    ax.set_xticks(list(x_nbbench) + list(x_pretrained) + list(x_sft))
    ax.set_xticklabels(models_nbbench + models_pretrained + models_sft,
                       rotation=45, ha='right', fontsize=5)
    ax.grid(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {out_path}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    df_test  = pd.read_csv(TEST_CSV)
    y_test   = df_test['label'].values
    seqs_test = df_test['seq'].tolist()

    rows = []

    for size in MODEL_SIZES:
        base_name = ESM_BASE_NAMES[size]
        print(f"\n{'='*50}")
        print(f"モデルサイズ: {size}  ({base_name})")
        tokenizer = AutoTokenizer.from_pretrained(base_name)

        for label, dir_suffix in [('Pre-trained', f'sft_esm2_{size}_head_only_optuna'),
                                   ('SFT',        f'sft_esm2_{size}_optuna')]:
            model_dir = SFT_DIR / dir_suffix
            print(f"  {label}: {dir_suffix}")

            # エンコーダー読み込み
            enc = AutoModel.from_pretrained(str(model_dir / 'encoder')).to(device)

            # テスト埋め込み
            emb_test = get_mean_pooled(enc, tokenizer, seqs_test, device)
            del enc
            torch.cuda.empty_cache()

            # 予測
            y_pred = predict_tm(model_dir, emb_test)

            # ブートストラップ
            ci = bootstrap_ci(y_test, y_pred)
            print(f"    Spearman: {ci['spearman'][0]:.3f} "
                  f"[{ci['spearman'][1]:.3f}, {ci['spearman'][2]:.3f}]")

            row = {'model_type': label, 'model_size': size,
                   'model_label': f'{label}_{size.upper()}'}
            for m in METRICS:
                pt, lo, hi = ci[m]
                row[m]          = round(pt, 4)
                row[f'{m}_lo']  = round(lo, 4)
                row[f'{m}_hi']  = round(hi, 4)
            rows.append(row)

    df_ci = pd.DataFrame(rows)
    df_ci.to_csv(OUTPUT_DIR / 'tables' / 'bootstrap_ci.csv', index=False)
    print(f"\n結果保存: {OUTPUT_DIR}/tables/bootstrap_ci.csv")

    # ── サマリー表示 ─────────────────────────────────────────────────────
    print("\n=== Spearman ρ (点推定 [95%CI]) ===")
    for _, r in df_ci.iterrows():
        print(f"  {r['model_label']:22s}: "
              f"{r['spearman']:.3f} [{r['spearman_lo']:.3f}, {r['spearman_hi']:.3f}]")

    # ── Fig. 3b ──────────────────────────────────────────────────────────
    plot_fig3b(df_ci, OUTPUT_DIR / 'figure' / 'Fig3b_spearman_by_model.png')

    return df_ci

    return df_ci


if __name__ == '__main__':
    main()
