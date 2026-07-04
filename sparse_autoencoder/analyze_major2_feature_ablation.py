#!/usr/bin/env python3
"""
追試2: Feature ablation解析 (Reviewer 2 Major 2)

Ridge重みの|w|上位N個のSAE特徴をablation（MaxAbsScaler適用後の空間で0置換）し、
テストセット性能の変化を計測。ランダムablationと比較することで、予測が少数の
ロバストな特徴に駆動されていることを示す。

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_major2_feature_ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# ── パス設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

SPARSE_DIR = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_7e-2'
TM_DATA    = SCRIPT_DIR / '../data/nbbench/thermo-seq/vhh_thermo_seq.csv'
TRAIN_CSV  = SCRIPT_DIR / '../data/nbbench/thermo-seq/train.csv'
TEST_CSV   = SCRIPT_DIR / '../data/nbbench/thermo-seq/test.csv'
OUTPUT_DIR = SCRIPT_DIR / 'outputs/major2_feature_ablation'

LAYER        = 6
CV_FOLDS     = 10
RANDOM_STATE = 42

# ablation の N（0 = baseline、上位N個をゼロ化）
ABLATION_NS  = [0, 5, 10, 20, 50, 100, 200]
N_RANDOM_TRIALS = 100   # ランダムablationの試行数


# ── データ・モデル準備 ────────────────────────────────────────────────────────

def setup_dirs():
    (OUTPUT_DIR / 'figure').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)


def load_and_fit():
    """sparse activationsを読み込み、MaxAbsScaler + RidgeCVをfit。
    テスト用のスケール済み行列と重みを返す。"""
    print("データ読み込み中...")
    df = pd.read_csv(TM_DATA)
    sparse_proteins = torch.load(
        SPARSE_DIR / f'all_sequences_layer_{LAYER}_sparse_activations.pt', map_location='cpu'
    )
    y = df['label'].values

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    seq2i = {s: i for i, s in enumerate(df['seq'])}
    train_idx = np.array([seq2i[s] for s in train_df['seq']], dtype=int)
    test_idx  = np.array([seq2i[s] for s in test_df['seq']],  dtype=int)

    X = torch.stack([p.mean(dim=0) for p in sparse_proteins]).numpy()

    scaler = MaxAbsScaler()
    X_tr_sc = scaler.fit_transform(X[train_idx])
    X_te_sc = scaler.transform(X[test_idx])      # ablation はこの空間で行う
    y_tr = y[train_idx]
    y_te = y[test_idx]

    alphas = np.logspace(-1, 4, 30)
    ridge = RidgeCV(alphas=alphas, cv=CV_FOLDS, scoring='neg_root_mean_squared_error')
    ridge.fit(X_tr_sc, y_tr)
    print(f"Ridge α={ridge.alpha_:.4f}")

    return ridge, X_te_sc, y_te


# ── 評価 ─────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    return {
        'rmse':    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':     float(mean_absolute_error(y_true, y_pred)),
        'r2':      float(r2_score(y_true, y_pred)),
        'pearson': float(pearsonr(y_true, y_pred)[0]),
        'spearman':float(spearmanr(y_true, y_pred)[0]),
    }


def ablate_and_eval(ridge, X_te_sc, y_te, zero_indices):
    """指定インデックスの次元を0にしてRidge予測→性能評価"""
    X_abl = X_te_sc.copy()
    if len(zero_indices) > 0:
        X_abl[:, zero_indices] = 0.0
    y_pred = ridge.predict(X_abl)
    return evaluate(y_te, y_pred)


# ── ablation 実行 ─────────────────────────────────────────────────────────────

def run_ablation(ridge, X_te_sc, y_te):
    """
    3種のablationを実行:
      top-k-abs  : |w| 上位N個をゼロ化（正負混合）
      top-k-pos  : 正の重み上位N個をゼロ化
      top-k-neg  : 負の重み上位N個（最も負）をゼロ化
      random     : ランダムN個をゼロ化（N_RANDOM_TRIALS 回の平均±SD）
    """
    weights = ridge.coef_
    rank_by_abs = np.argsort(np.abs(weights))[::-1]            # |w| 降順
    rank_pos    = np.argsort(weights)[::-1]                    # 正方向降順
    rank_neg    = np.argsort(weights)                          # 負方向降順

    rng = np.random.default_rng(RANDOM_STATE)
    n_features = X_te_sc.shape[1]

    rows = []

    for n in ABLATION_NS:
        # ── top-k-abs ──
        m = ablate_and_eval(ridge, X_te_sc, y_te, rank_by_abs[:n])
        rows.append({'n': n, 'type': 'top-k-abs', **m})

        # ── top-k-pos ──
        m = ablate_and_eval(ridge, X_te_sc, y_te, rank_pos[:n])
        rows.append({'n': n, 'type': 'top-k-pos', **m})

        # ── top-k-neg ──
        m = ablate_and_eval(ridge, X_te_sc, y_te, rank_neg[:n])
        rows.append({'n': n, 'type': 'top-k-neg', **m})

        print(f"  n={n:3d} | abs: RMSE={rows[-3]['rmse']:.3f} | "
              f"pos: RMSE={rows[-2]['rmse']:.3f} | neg: RMSE={rows[-1]['rmse']:.3f}")

        # ── random ──
        if n == 0:
            rows.append({'n': n, 'type': 'random_mean', **m,
                         'rmse_sd':0., 'mae_sd':0., 'r2_sd':0.,
                         'pearson_sd':0., 'spearman_sd':0.})
            continue

        rand_metrics_list = {k: [] for k in ['rmse','mae','r2','pearson','spearman']}
        for _ in range(N_RANDOM_TRIALS):
            rand_idx = rng.choice(n_features, size=n, replace=False)
            mi = ablate_and_eval(ridge, X_te_sc, y_te, rand_idx)
            for k in rand_metrics_list:
                rand_metrics_list[k].append(mi[k])

        rand_row = {'n': n, 'type': 'random_mean'}
        for k in ['rmse','mae','r2','pearson','spearman']:
            arr = np.array(rand_metrics_list[k])
            rand_row[k]         = float(arr.mean())
            rand_row[f'{k}_sd'] = float(arr.std())
        rows.append(rand_row)
        print(f"         | rand: RMSE={rand_row['rmse']:.3f}±{rand_row['rmse_sd']:.3f}")

    df = pd.DataFrame(rows)
    for col in ['rmse_sd','mae_sd','r2_sd','pearson_sd','spearman_sd']:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ── 可視化 ────────────────────────────────────────────────────────────────────

METRICS_LABEL = {
    'rmse':     'RMSE (°C)',
    'mae':      'MAE (°C)',
    'r2':       'R²',
    'pearson':  'Pearson r',
    'spearman': 'Spearman ρ',
}

STYLE = {
    'top-k-abs': dict(color='#d62728', marker='o', ls='-',  label='Top-k |w| (abs)'),
    'top-k-pos': dict(color='#e6550d', marker='^', ls='--', label='Top-k pos w'),
    'top-k-neg': dict(color='#3182bd', marker='v', ls='--', label='Top-k neg w'),
    'random_mean': dict(color='gray',  marker='s', ls=':',  label=f'Random (mean±SD, n={N_RANDOM_TRIALS})'),
}


def plot_ablation(df_results, metric):
    """4条件の折れ線グラフ（1指標）"""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for abl_type, style in STYLE.items():
        sub = df_results[df_results['type'] == abl_type].sort_values('n')
        ax.plot(sub['n'], sub[metric], marker=style['marker'], ls=style['ls'],
                color=style['color'], linewidth=1.5, markersize=5,
                label=style['label'], zorder=3)
        sd_col = f'{metric}_sd'
        if abl_type == 'random_mean' and sd_col in sub.columns and sub[sd_col].notna().any():
            ax.fill_between(sub['n'], sub[metric]-sub[sd_col], sub[metric]+sub[sd_col],
                            color=style['color'], alpha=0.15, zorder=1)

    ax.set_xlabel('Number of ablated features', fontsize=11)
    ax.set_ylabel(METRICS_LABEL.get(metric, metric), fontsize=11)
    ax.legend(fontsize=7.5, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = OUTPUT_DIR / f'figure/ablation_{metric}.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {path}")


def plot_ablation_all(df_results):
    """全指標を2×3サブプロット（まとめ図）"""
    metrics = ['rmse', 'mae', 'r2', 'pearson', 'spearman']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        for abl_type, style in STYLE.items():
            sub = df_results[df_results['type'] == abl_type].sort_values('n')
            ax.plot(sub['n'], sub[metric], marker=style['marker'], ls=style['ls'],
                    color=style['color'], linewidth=1.2, markersize=4,
                    label=style['label'])
            sd_col = f'{metric}_sd'
            if abl_type == 'random_mean' and sd_col in sub.columns and sub[sd_col].notna().any():
                ax.fill_between(sub['n'], sub[metric]-sub[sd_col], sub[metric]+sub[sd_col],
                                color=style['color'], alpha=0.15)
        ax.set_title(METRICS_LABEL.get(metric, metric), fontsize=10)
        ax.set_xlabel('# ablated features', fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=6.5)

    axes_flat[-1].set_visible(False)
    plt.suptitle(f'Feature ablation: top-k vs random ({N_RANDOM_TRIALS} trials)\n'
                 'SAE sparse features — MaxAbsScaler space', fontsize=11)
    plt.tight_layout()
    path = OUTPUT_DIR / 'figure/ablation_all_metrics.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  まとめ図保存: {path}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    setup_dirs()

    ridge, X_te_sc, y_te = load_and_fit()

    weights = ridge.coef_
    rank_by_abs = np.argsort(np.abs(weights))[::-1]
    print(f"\n重みの絶対値上位5: "
          f"{[(int(rank_by_abs[i]), round(float(weights[rank_by_abs[i]]),4)) for i in range(5)]}")
    print(f"全{len(weights)}特徴中 |w|>0.1: {(np.abs(weights)>0.1).sum()}個\n")

    print("=== Ablation実行 ===")
    df_results = run_ablation(ridge, X_te_sc, y_te)

    # テーブル保存
    df_results.to_csv(OUTPUT_DIR / 'tables/ablation_results.csv', index=False)
    print(f"\nテーブル保存: {OUTPUT_DIR}/tables/ablation_results.csv")

    # 見やすい成形テーブルも保存
    rows_abs  = df_results[df_results['type']=='top-k-abs'].set_index('n')
    rows_pos  = df_results[df_results['type']=='top-k-pos'].set_index('n')
    rows_neg  = df_results[df_results['type']=='top-k-neg'].set_index('n')
    rand_rows = df_results[df_results['type']=='random_mean'].set_index('n')
    summary_rows = []
    for n in ABLATION_NS:
        row = {'n_ablated': n}
        for m in ['rmse','mae','r2','pearson','spearman']:
            row[f'abs_{m}'] = round(rows_abs.loc[n, m], 4) if n in rows_abs.index else np.nan
            row[f'pos_{m}'] = round(rows_pos.loc[n, m], 4) if n in rows_pos.index else np.nan
            row[f'neg_{m}'] = round(rows_neg.loc[n, m], 4) if n in rows_neg.index else np.nan
            if n in rand_rows.index:
                row[f'rand_{m}_mean'] = round(rand_rows.loc[n, m], 4)
                row[f'rand_{m}_sd']   = round(rand_rows.loc[n, f'{m}_sd'], 4) if f'{m}_sd' in rand_rows.columns else np.nan
            else:
                row[f'rand_{m}_mean'] = np.nan
                row[f'rand_{m}_sd']   = np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'tables/ablation_summary.csv', index=False)

    # 可視化
    print("\n=== 図の生成 ===")
    for metric in ['rmse', 'mae', 'r2', 'pearson', 'spearman']:
        plot_ablation(df_results, metric)
    plot_ablation_all(df_results)

    print(f"\n完了。出力先: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
