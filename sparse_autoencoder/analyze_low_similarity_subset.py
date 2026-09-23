#!/usr/bin/env python3
"""
低配列類似度サブセットでの発火プロファイル比較

生成物（論文に採用された図のみ）:
    figure/FigS9_firing_profile_positive.png  -- 正の重みTop10特徴（Fig. S4上位と同一特徴）
    figure/FigS10_firing_profile_negative.png -- 負の重みTop10特徴（Fig. S4下位と同一特徴）

手順:
    1. analyze_sae.py の既存パイプライン（train_and_evaluate_model）でRidge回帰を再現し、
       Fig. S4 と同一の正/負Top10特徴（get_top_bottom_features）を取得する。
    2. テスト147配列についてtrain 522配列との最大配列同一性を算出
       （sequence_similarity_validation.ipynb と同一のPairwiseAligner設定）。
    3. 同一性80%を閾値に低/高類似度サブセットに分割し、各特徴のAHo位置別平均発火
       プロファイルを比較（analyze_sae.py の build_aho_activation_matrix を再利用）。

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_low_similarity_subset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from Bio.Align import PairwiseAligner

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_sae import (
    load_data,
    mean_pool_proteins,
    train_and_evaluate_model,
    get_top_bottom_features,
    build_aho_activation_matrix,
)

# ── パス設定 ──────────────────────────────────────────────────────────────────
DENSE_DIR  = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_esm_sft_8m'
SPARSE_DIR = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_7e-2'
TM_DATA    = SCRIPT_DIR / '../data/nbbench/thermo-seq/vhh_thermo_seq.csv'
TRAIN_CSV  = SCRIPT_DIR / '../data/nbbench/thermo-seq/train.csv'
TEST_CSV   = SCRIPT_DIR / '../data/nbbench/thermo-seq/test.csv'
OUTPUT_DIR = SCRIPT_DIR / 'outputs/low_similarity_subset'

LAYER         = 6
CV_FOLDS      = 10
RANDOM_STATE  = 42
SIM_THRESHOLD = 0.80   # 低類似度の閾値
TOP_N         = 10     # 正/負それぞれの重み上位N個（Fig. S4と同一）
AHO_LEN       = 149


def compute_max_identity(test_seqs, train_seqs):
    """
    各test配列についてtrain配列との最大グローバルアライメント同一性を計算。
    identity = matches / alignment_length（gap含む）。
    sequence_similarity_validation.ipynb と同一の指標・パラメータ。
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    max_ids = np.zeros(len(test_seqs))
    for i, tseq in enumerate(test_seqs):
        best = 0.0
        for trseq in train_seqs:
            aln = next(iter(aligner.align(tseq, trseq)))
            ident = aln.score / aln.shape[1]
            if ident > best:
                best = ident
        max_ids[i] = best
        if (i + 1) % 20 == 0 or (i + 1) == len(test_seqs):
            print(f"  配列同一性計算: {i + 1}/{len(test_seqs)} 完了")
    return max_ids


def plot_profile_grid(sparse_low, df_low, sparse_high, df_high,
                       feat_indices, weights, group_label, out_path):
    """低/高類似度サブセット間の発火プロファイルを1特徴1パネルのグリッド図にまとめる。"""
    n = len(feat_indices)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(AHO_LEN) + 1

    for rank, (ax, feat_idx) in enumerate(zip(axes, feat_indices), start=1):
        M_low, _  = build_aho_activation_matrix(sparse_low,  df_low,  feat_idx, aho_len=AHO_LEN)
        M_high, _ = build_aho_activation_matrix(sparse_high, df_high, feat_idx, aho_len=AHO_LEN)
        mean_low, mean_high = M_low.mean(axis=0), M_high.mean(axis=0)
        r, _ = pearsonr(mean_low, mean_high)

        ax.fill_between(x, mean_high, alpha=0.4, color='steelblue', label='High-sim')
        ax.fill_between(x, mean_low,  alpha=0.4, color='crimson',   label='Low-sim')
        ax.plot(x, mean_high, color='steelblue', linewidth=1)
        ax.plot(x, mean_low,  color='crimson',   linewidth=1)
        ax.set_title(f'#{rank} Feat {feat_idx} (w={weights[feat_idx]:+.3f}, r={r:.2f})', fontsize=8)
        ax.tick_params(axis='both', labelsize=6)

    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Firing profile: low- vs high-similarity subset ({group_label}-weight top{n})', fontsize=10)
    plt.tight_layout()
    (out_path.parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {out_path}")


def main():
    # 1. データ読み込み + Ridge回帰（既存パイプラインを再利用、Fig. S4と同一手続き）
    df, y, dense_proteins, sparse_proteins = load_data(DENSE_DIR, SPARSE_DIR, TM_DATA, LAYER)
    X_dense  = mean_pool_proteins(dense_proteins).numpy()
    X_sparse = mean_pool_proteins(sparse_proteins).numpy()

    results = train_and_evaluate_model(
        X_dense, X_sparse, y, df, str(TM_DATA), CV_FOLDS, str(OUTPUT_DIR),
        random_state=RANDOM_STATE,
    )
    sparse_weights = results['ridgecv_sparse'].coef_
    pos_idx, neg_idx = get_top_bottom_features(sparse_weights, n=TOP_N)
    print(f"\n正の重みTop{TOP_N}: {pos_idx.tolist()}")
    print(f"負の重みTop{TOP_N}: {neg_idx.tolist()}")

    # 2. train/testインデックス（train_and_evaluate_model内と同一ロジック）
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    seq_to_idx = {seq: i for i, seq in enumerate(df['seq'])}
    test_idx = np.array([seq_to_idx[s] for s in test_df['seq']], dtype=int)

    # 3. test vs train 配列同一性 → 低/高類似度サブセット分割
    print("\ntest vs train 配列同一性計算中...")
    max_ids = compute_max_identity(test_df['seq'].tolist(), train_df['seq'].tolist())
    low_mask  = max_ids <= SIM_THRESHOLD
    high_mask = ~low_mask
    print(f"低類似度 (<={int(SIM_THRESHOLD*100)}%): n={low_mask.sum()}, "
          f"高類似度 (>{int(SIM_THRESHOLD*100)}%): n={high_mask.sum()}")

    low_test_idx  = test_idx[low_mask]
    high_test_idx = test_idx[high_mask]
    df_low  = df.iloc[low_test_idx].reset_index(drop=True)
    df_high = df.iloc[high_test_idx].reset_index(drop=True)
    sparse_low  = [sparse_proteins[i] for i in low_test_idx]
    sparse_high = [sparse_proteins[i] for i in high_test_idx]

    # 4. Fig S9 (正の重みTop10) / Fig S10 (負の重みTop10)
    print("\nFig S9 (正の重みTop10) 作成中...")
    plot_profile_grid(sparse_low, df_low, sparse_high, df_high, pos_idx, sparse_weights,
                       'positive', OUTPUT_DIR / 'figure' / 'FigS9_firing_profile_positive.png')

    print("\nFig S10 (負の重みTop10) 作成中...")
    plot_profile_grid(sparse_low, df_low, sparse_high, df_high, neg_idx, sparse_weights,
                       'negative', OUTPUT_DIR / 'figure' / 'FigS10_firing_profile_negative.png')

    print(f"\n完了。出力先: {OUTPUT_DIR}/figure/")


if __name__ == '__main__':
    main()
