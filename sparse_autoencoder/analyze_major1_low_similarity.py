#!/usr/bin/env python3
"""
追試1: 低配列類似度サブセット解析 (Reviewer 2 Major 1)

テスト147配列に対してtrain 522配列との最大pairwise配列同一性を算出し、
80%以下の低類似度サブセットを抽出。主要SAE特徴（|w_k|>=0.05）の
発火パターンを高類似度群と比較し、配列類似度に依存せず共通AHo位置で
発火することを示す。

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_major1_low_similarity.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from Bio.Align import PairwiseAligner

# ── パス設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

DENSE_DIR   = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_esm_sft_8m'
SPARSE_DIR  = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_7e-2'
TM_DATA     = SCRIPT_DIR / '../data/nbbench/thermo-seq/vhh_thermo_seq.csv'
TRAIN_CSV   = SCRIPT_DIR / '../data/nbbench/thermo-seq/train.csv'
TEST_CSV    = SCRIPT_DIR / '../data/nbbench/thermo-seq/test.csv'
OUTPUT_DIR  = SCRIPT_DIR / 'outputs/major1_low_similarity'

LAYER           = 6
CV_FOLDS        = 10
RANDOM_STATE    = 42
SIM_THRESHOLD   = 0.80   # 低類似度の閾値
TOP_N_FEATURES  = 10     # |w|上位N個を主要特徴として解析
AHO_LEN         = 149    # AHo列数


# ── ユーティリティ ────────────────────────────────────────────────────────────

def setup_dirs():
    (OUTPUT_DIR / 'figure').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)


def load_activations():
    print("データ読み込み中...")
    df = pd.read_csv(TM_DATA)
    dense_proteins = torch.load(
        DENSE_DIR / f'all_sequences_layer_{LAYER}_activations.pt', map_location='cpu'
    )
    sparse_proteins = torch.load(
        SPARSE_DIR / f'all_sequences_layer_{LAYER}_sparse_activations.pt', map_location='cpu'
    )
    print(f"  vhh_thermo_seq: {len(df)}配列, dense: {len(dense_proteins)}, sparse: {len(sparse_proteins)}")
    return df, dense_proteins, sparse_proteins


def get_split_indices(df):
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    seq_to_idx = {seq: i for i, seq in enumerate(df['seq'])}
    train_idx = np.array([seq_to_idx[s] for s in train_df['seq']], dtype=int)
    test_idx  = np.array([seq_to_idx[s] for s in test_df['seq']],  dtype=int)
    return train_idx, test_idx, train_df, test_df


def mean_pool(protein_list):
    return torch.stack([p.mean(dim=0) for p in protein_list]).numpy()


def fit_ridge(X_sparse_pool, y, train_idx, test_idx):
    """MaxAbsScaler + RidgeCVでモデルをfit（analyze_sae.py の実装を踏襲）"""
    X_tr = X_sparse_pool[train_idx]
    X_te = X_sparse_pool[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    scaler = MaxAbsScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    alphas = np.logspace(-1, 4, 30)
    ridge = RidgeCV(alphas=alphas, cv=CV_FOLDS, scoring='neg_root_mean_squared_error')
    ridge.fit(X_tr_sc, y_tr)

    y_pred = ridge.predict(X_te_sc)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)
    pr     = pearsonr(y_te, y_pred)[0]
    sp     = spearmanr(y_te, y_pred)[0]
    print(f"Ridge(sparse): RMSE={rmse:.3f}, R²={r2:.3f}, Pearson={pr:.3f}, Spearman={sp:.3f}")
    return ridge, scaler, y_pred, y_te


# ── 配列類似度計算 ────────────────────────────────────────────────────────────

def compute_max_identity(test_seqs, train_seqs):
    """
    各test配列についてtrain配列との最大グローバルアライメント同一性を計算。
    identity = matches / alignment_length（gap含む）
    sequence_similarity_validation.ipynb と同一の指標・方法。
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score    = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score    = 0
    aligner.extend_gap_score  = 0

    max_ids = []
    n = len(test_seqs)
    for i, tseq in enumerate(test_seqs):
        best = 0.0
        for trseq in train_seqs:
            aln     = next(iter(aligner.align(tseq, trseq)))
            aln_len = aln.shape[1]
            matches = aln.score   # match=1, mismatch/gap=0 なので score = matches
            ident   = matches / aln_len if aln_len > 0 else 0.0
            if ident > best:
                best = ident
        max_ids.append(best)
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  類似度計算: {i+1}/{n} 完了")
    return np.array(max_ids)


# ── AHo発火パターン ───────────────────────────────────────────────────────────

def build_ungapped_to_aho_map(seq_aho):
    """analyze_sae.py と同一実装"""
    mapping = []
    for i, ch in enumerate(seq_aho):
        if ch != '-':
            mapping.append(i + 1)  # 1-based AHo位置
    return mapping


def build_aho_activation_matrix(sparse_proteins, df_subset, feature_idx, aho_len=AHO_LEN):
    """
    sparse_proteins: 全764配列のsparse activationリスト
    df_subset: dfのサブセット（indexがsparse_proteinsのindexに対応）
    """
    rows = []
    for _, row in df_subset.iterrows():
        global_idx = row['_global_idx']
        activations = sparse_proteins[global_idx][:, feature_idx].numpy()
        seq_aho = row['sequence_aho']
        u2a = build_ungapped_to_aho_map(seq_aho)
        aho_act = np.zeros(aho_len)
        L = min(len(activations), len(u2a))
        for u_idx in range(L):
            aho_pos = u2a[u_idx] - 1  # 0-based
            if aho_pos < aho_len:
                aho_act[aho_pos] = activations[u_idx]
        rows.append(aho_act)
    return np.stack(rows, axis=0)  # (N, aho_len)


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_similarity_distribution(max_ids_test, threshold=SIM_THRESHOLD):
    """test配列の最大類似度ヒストグラム"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(max_ids_test * 100, bins=20, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(threshold * 100, color='crimson', linestyle='--', linewidth=1.5,
               label=f'Threshold {int(threshold*100)}%')
    n_low  = np.sum(max_ids_test <= threshold)
    n_high = np.sum(max_ids_test > threshold)
    ax.set_xlabel('Max sequence identity to train (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.97,
            f'Low (<={int(threshold*100)}%): n={n_low}\nHigh (>{int(threshold*100)}%): n={n_high}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    path = OUTPUT_DIR / 'figure/similarity_distribution.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {path}")


def plot_similarity_vs_pred_error(max_ids_test, y_obs, y_pred, threshold=SIM_THRESHOLD):
    """配列類似度 vs 予測誤差の散布図"""
    abs_err = np.abs(y_obs - y_pred)
    low_mask = max_ids_test <= threshold

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(max_ids_test[~low_mask] * 100, abs_err[~low_mask],
               c='steelblue', s=20, alpha=0.7, label=f'High (>{int(threshold*100)}%)')
    ax.scatter(max_ids_test[low_mask] * 100, abs_err[low_mask],
               c='crimson', s=30, alpha=0.9, marker='D', label=f'Low (<={int(threshold*100)}%)')
    ax.set_xlabel('Max sequence identity to train (%)', fontsize=11)
    ax.set_ylabel('|Tm_pred - Tm_obs| (°C)', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / 'figure/similarity_vs_pred_error.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {path}")


def plot_firing_profile_comparison(low_profiles, high_profiles, feature_idx, weight_val,
                                   top_n_pos=10):
    """
    低類似度群 vs 高類似度群の mean per-AHo-position 発火プロファイル比較。
    上位10位のAHo位置をハイライト。
    """
    # 高類似度群の上位位置を基準に整列
    mean_high = np.mean(high_profiles, axis=0)
    mean_low  = np.mean(low_profiles,  axis=0)
    top_pos   = np.argsort(mean_high)[-top_n_pos:][::-1]

    r, _ = pearsonr(mean_high, mean_low)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(AHO_LEN) + 1
    ax.fill_between(x, mean_high, alpha=0.4, color='steelblue', label='High-sim group')
    ax.fill_between(x, mean_low,  alpha=0.4, color='crimson',   label='Low-sim group')
    ax.plot(x, mean_high, color='steelblue', linewidth=1)
    ax.plot(x, mean_low,  color='crimson',   linewidth=1)

    for pos in top_pos:
        ax.axvline(pos + 1, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    sign = '+' if weight_val >= 0 else ''
    ax.set_title(f'Feature {feature_idx}  (w={sign}{weight_val:.3f}, Pearson r={r:.3f})', fontsize=10)
    ax.set_xlabel('AHo position', fontsize=10)
    ax.set_ylabel('Mean activation', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / f'figure/firing_profile_feature{feature_idx}.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    return r


def plot_similarity_vs_profile_corr(max_ids_test, profile_corrs_per_seq, feature_idx):
    """
    各testシーケンスの max_identity vs train平均プロファイルとのPearson相関。
    NaN（発火なし）はプロットから除外する。
    """
    valid = ~np.isnan(profile_corrs_per_seq)
    low_mask = (max_ids_test <= SIM_THRESHOLD) & valid
    hi_mask  = (max_ids_test  > SIM_THRESHOLD) & valid

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(max_ids_test[hi_mask] * 100, profile_corrs_per_seq[hi_mask],
               c='steelblue', s=15, alpha=0.7, label='High-sim')
    ax.scatter(max_ids_test[low_mask] * 100, profile_corrs_per_seq[low_mask],
               c='crimson', s=25, alpha=0.9, marker='D', label='Low-sim')
    ax.axvline(SIM_THRESHOLD * 100, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Max sequence identity to train (%)', fontsize=11)
    ax.set_ylabel('Pearson r (sequence profile vs train mean)', fontsize=10)
    ax.set_title(f'Feature {feature_idx}', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / f'figure/sim_vs_profile_corr_feature{feature_idx}.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()


def plot_aho_heatmap_low_sim(sparse_proteins, df_low, feature_idx, weight_val):
    """低類似度配列のAHo発火ヒートマップ"""
    M = build_aho_activation_matrix(sparse_proteins, df_low, feature_idx)
    order = np.argsort(df_low['label'].values)[::-1]  # Tm降順
    M_sorted = M[order]
    labels_sorted = df_low['label'].values[order]

    heatmap_df = pd.DataFrame(
        M_sorted,
        columns=[f'AHO{j+1}' for j in range(AHO_LEN)],
        index=[f'Tm={t:.1f}°C' for t in labels_sorted]
    )

    fig, ax = plt.subplots(figsize=(7, max(2, len(df_low) * 0.18 + 1)))
    cmap = 'Reds' if weight_val >= 0 else 'Blues'
    sns.heatmap(heatmap_df, ax=ax, cmap=cmap,
                cbar_kws={'label': 'Activation', 'shrink': 0.6},
                yticklabels=True, xticklabels=False)

    x_ticks = list(range(0, AHO_LEN, 10))
    ax.set_xticks([x + 0.5 for x in x_ticks])
    ax.set_xticklabels([str(x + 1) for x in x_ticks], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('AHo position', fontsize=10)
    ax.set_ylabel('Low-similarity sequences (sorted by Tm)', fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    sign = '+' if weight_val >= 0 else ''
    ax.set_title(f'Feature {feature_idx}  (w={sign}{weight_val:.3f})  — Low-similarity subset', fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / f'figure/aho_heatmap_lowsim_feature{feature_idx}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ヒートマップ保存: {path}")


def plot_summary_profile_corrs(feature_summary):
    """
    各主要特徴について、低類似度群 vs 高類似度群のプロファイル相関 (bar chart)
    """
    feat_ids  = [r['feature_idx']      for r in feature_summary]
    prof_corrs= [r['profile_pearson_r'] for r in feature_summary]
    weights   = [r['weight']            for r in feature_summary]
    colors    = ['crimson' if w > 0 else 'steelblue' for w in weights]

    fig, ax = plt.subplots(figsize=(max(6, len(feat_ids) * 0.5 + 1), 3.5))
    bars = ax.bar(range(len(feat_ids)), prof_corrs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(feat_ids)))
    ax.set_xticklabels([str(f) for f in feat_ids], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('SAE Feature index', fontsize=11)
    ax.set_ylabel('Pearson r (low-sim vs high-sim profile)', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Firing profile similarity: low-sim group vs high-sim group\nfor top SAE features (|w|≥0.05)', fontsize=10)
    plt.tight_layout()
    path = OUTPUT_DIR / 'figure/summary_profile_corrs.png'
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {path}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    setup_dirs()

    # 1. データ・活性化データ読み込み
    df, dense_proteins, sparse_proteins = load_activations()
    train_idx, test_idx, train_df, test_df = get_split_indices(df)

    y = df['label'].values

    # vhh_thermo_seq.csv上のグローバルインデックスを付与（後でdropできるよう保存）
    df['_global_idx'] = np.arange(len(df))

    # 2. mean-pool sparse activations → Ridge fit
    print("\nMean-pool + Ridge学習...")
    X_sparse_pool = mean_pool(sparse_proteins)
    ridge, scaler, y_pred_test, y_obs_test = fit_ridge(X_sparse_pool, y, train_idx, test_idx)
    sparse_weights = ridge.coef_

    # testサブセットのdf
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    df_test['Tm_obs']  = y_obs_test
    df_test['Tm_pred'] = y_pred_test

    # 3. 配列類似度計算（test vs train）
    print("\ntest vs train 配列同一性計算中...")
    test_seqs  = test_df['seq'].tolist()
    train_seqs = train_df['seq'].tolist()
    max_ids = compute_max_identity(test_seqs, train_seqs)

    df_test['max_identity'] = max_ids

    # 4. 低類似度サブセット抽出
    low_mask  = max_ids <= SIM_THRESHOLD
    high_mask = ~low_mask
    n_low  = low_mask.sum()
    n_high = high_mask.sum()
    print(f"\n低類似度 (≤{int(SIM_THRESHOLD*100)}%): n={n_low}, 高類似度: n={n_high}")

    df_low  = df_test[low_mask].copy().reset_index(drop=True)
    df_high = df_test[high_mask].copy().reset_index(drop=True)

    # サブセットテーブル保存
    cols_out = ['id', 'seq', 'max_identity', 'Tm_obs', 'Tm_pred']
    df_low_out = df_low[cols_out].copy()
    df_low_out['max_identity'] = (df_low_out['max_identity'] * 100).round(2)
    df_low_out.to_csv(OUTPUT_DIR / 'tables/low_similarity_subset.csv', index=False)

    df_test_out = df_test[cols_out].copy()
    df_test_out['max_identity'] = (df_test_out['max_identity'] * 100).round(2)
    df_test_out.to_csv(OUTPUT_DIR / 'tables/all_test_with_similarity.csv', index=False)
    print(f"  テーブル保存: {OUTPUT_DIR}/tables/")

    # 5. 図: 類似度分布 / 予測誤差
    print("\n類似度分布図を作成中...")
    plot_similarity_distribution(max_ids)
    plot_similarity_vs_pred_error(max_ids, y_obs_test, y_pred_test)

    # 6. 主要特徴の発火パターン解析（|w|上位TOP_N_FEATURES個）
    top_feat_order = np.argsort(np.abs(sparse_weights))[-TOP_N_FEATURES:][::-1]
    top_feat_idx   = top_feat_order
    top_feat_w     = sparse_weights[top_feat_order]
    print(f"\n主要特徴 (|w|上位{TOP_N_FEATURES}個): min|w|={np.abs(top_feat_w).min():.4f}")
    print(f"  インデックス: {top_feat_idx.tolist()}")
    print(f"  重み: {[round(float(x),4) for x in top_feat_w]}")

    feature_summary = []

    # vhh_thermo_seq.csv上のインデックスが必要なので df_low/df_high の _global_idx を使う
    # df_testのインデックスをdf全体のglobal_idxに変換するためのマッピング
    for feat_idx, feat_w in zip(top_feat_idx, top_feat_w):
        print(f"\n  Feature {feat_idx} (w={feat_w:+.4f}) 解析中...")

        # 低類似度群のAHo発火行列
        low_aho  = build_aho_activation_matrix(sparse_proteins, df_low,  feat_idx)   # (n_low, AHO_LEN)
        high_aho = build_aho_activation_matrix(sparse_proteins, df_high, feat_idx)   # (n_high, AHO_LEN)

        mean_low  = low_aho.mean(axis=0)
        mean_high = high_aho.mean(axis=0)

        # プロファイル間Pearson相関
        r_profile, _ = pearsonr(mean_low, mean_high)
        print(f"    低類似度 vs 高類似度 プロファイルPearson r={r_profile:.4f}")

        # 各testシーケンスのプロファイル vs 全train平均との比較
        # train上の全配列で mean profile を計算
        df_train_sub = df.iloc[train_idx].copy().reset_index(drop=True)
        train_aho = build_aho_activation_matrix(sparse_proteins, df_train_sub, feat_idx)
        mean_train = train_aho.mean(axis=0)

        # 全testシーケンスのper-sequence profilre vs train mean
        all_test_aho = build_aho_activation_matrix(sparse_proteins, df_test, feat_idx)
        def safe_pearsonr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return np.nan
            return pearsonr(a, b)[0]

        per_seq_corr = np.array([
            safe_pearsonr(all_test_aho[i], mean_train)
            for i in range(len(df_test))
        ])

        # 図: 低類似度群 vs 高類似度群のプロファイル比較
        r_val = plot_firing_profile_comparison(low_aho, high_aho, feat_idx, feat_w)

        # 図: 各testシーケンスの max_identity vs プロファイル相関
        plot_similarity_vs_profile_corr(max_ids, per_seq_corr, feat_idx)

        # 低類似度サブセットのAHoヒートマップ（n_low>=2のとき描画）
        if n_low >= 2:
            plot_aho_heatmap_low_sim(sparse_proteins, df_low, feat_idx, feat_w)

        # 低類似度群の上位5 AHo発火位置
        top5_aho_low = np.argsort(mean_low)[-5:][::-1] + 1  # 1-based
        top5_act_low = mean_low[top5_aho_low - 1]
        # 高類似度群の上位5 AHo発火位置
        top5_aho_high = np.argsort(mean_high)[-5:][::-1] + 1
        top5_act_high = mean_high[top5_aho_high - 1]

        print(f"    低類似度群 top-5 AHo位置: {top5_aho_low.tolist()}")
        print(f"    高類似度群 top-5 AHo位置: {top5_aho_high.tolist()}")

        feature_summary.append({
            'feature_idx':       int(feat_idx),
            'weight':            float(feat_w),
            'profile_pearson_r': float(r_profile),
            'mean_low_sim_max_activation':  float(mean_low.max()),
            'mean_high_sim_max_activation': float(mean_high.max()),
            'top5_aho_low':  ','.join(map(str, top5_aho_low.tolist())),
            'top5_aho_high': ','.join(map(str, top5_aho_high.tolist())),
        })

    # 7. サマリー図
    print("\nサマリー図作成中...")
    if feature_summary:
        plot_summary_profile_corrs(feature_summary)

    # サマリーテーブル保存
    df_summary = pd.DataFrame(feature_summary)
    df_summary.to_csv(OUTPUT_DIR / 'tables/feature_profile_summary.csv', index=False)

    # 8. 低類似度群の予測性能（参考）
    print("\n=== 低類似度サブセット (≤80%) の予測性能 ===")
    if n_low >= 5:
        rmse_low = np.sqrt(mean_squared_error(df_low['Tm_obs'], df_low['Tm_pred']))
        r2_low   = r2_score(df_low['Tm_obs'], df_low['Tm_pred'])
        pr_low   = pearsonr(df_low['Tm_obs'], df_low['Tm_pred'])[0]
        sp_low   = spearmanr(df_low['Tm_obs'], df_low['Tm_pred'])[0]
        print(f"  n={n_low}: RMSE={rmse_low:.3f}, R²={r2_low:.3f}, Pearson={pr_low:.3f}, Spearman={sp_low:.3f}")
    else:
        print(f"  n={n_low} — サンプル数が少ないため統計指標は省略（可視化を主軸とする）")

    print(f"\n=== 高類似度サブセット (>80%) の予測性能 ===")
    if n_high >= 5:
        rmse_high = np.sqrt(mean_squared_error(df_high['Tm_obs'], df_high['Tm_pred']))
        r2_high   = r2_score(df_high['Tm_obs'], df_high['Tm_pred'])
        pr_high   = pearsonr(df_high['Tm_obs'], df_high['Tm_pred'])[0]
        sp_high   = spearmanr(df_high['Tm_obs'], df_high['Tm_pred'])[0]
        print(f"  n={n_high}: RMSE={rmse_high:.3f}, R²={r2_high:.3f}, Pearson={pr_high:.3f}, Spearman={sp_high:.3f}")

    print(f"\n完了。出力先: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
