#!/usr/bin/env python3
"""
SAE特徴のablation解析（テストセット、bootstrap 95%信頼区間付き）

Ridge回帰の|w|上位N個のSAE特徴をablation（MaxAbsScaler適用後の空間で0置換）し、
テストセット（n=147）での性能変化を、同数のランダムablationと比較する。
top-|w| ablationは性能を急激に悪化させる一方、ランダムablationは同数を除去しても
性能がほぼ変化しないという非対称性から、予測が少数のロバストな特徴に駆動されている
ことを示す。

生成物（論文に採用された図表のみ）:
    figure/FigS4_feature_ablation.png   -- (a) RMSE, (b) MAE, (c) R² を
                                            top-|w| ablation vs random ablation で比較
                                            （bootstrap 95%信頼区間帯付き）
    tables/ablation_test_bootstrap_ci.csv -- 全指標（RMSE/MAE/R²/Pearson/Spearman）×
                                              全ablation数 × 2条件の点推定+95%CI

信頼区間の算出方針:
    - top-|w| ablation: ablationする特徴集合はNで一意に決まるため、予測値は1本に定まる。
      その予測値に対しテストセット(n=147)をpercentile bootstrapで1000回リサンプリングし、
      「小標本(n=147)の抽出に由来する不確実性」のみを表すCIを計算する。
    - random ablation: ablationする特徴の組み合わせ自体が乱数のため、「特徴選択のブレ」と
      「標本抽出のブレ」を同時に乱数化するnested bootstrapで95%CIを計算する（試行ごとに
      ランダムなN個を選択→その予測値に対し147サンプルを復元抽出）。
      点推定（中心線）は元のテストセットで、特徴選択のブレのみを1000試行平均した値。

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_feature_ablation.py
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
OUTPUT_DIR = SCRIPT_DIR / 'outputs/feature_ablation'

LAYER        = 6
CV_FOLDS     = 10
RANDOM_STATE = 42

ABLATION_NS         = [0, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
N_CI_BOOTSTRAP       = 1000   # top-|w|: テストセットのpercentile bootstrap回数
N_CI_RANDOM_TRIALS   = 1000   # random: 特徴選択+テストセットのnested bootstrap試行数

METRICS = ['rmse', 'mae', 'r2', 'pearson', 'spearman']
METRICS_LABEL = {'rmse': 'RMSE (°C)', 'mae': 'MAE (°C)', 'r2': 'R²'}
FIGURE_METRICS = ['rmse', 'mae', 'r2']   # Fig S4 の3パネル(a/b/c)

STYLE = {
    'top-k-abs':   dict(color='#d62728', marker='o', ls='-', label='Top-|w| ablation'),
    'random_mean': dict(color='gray',    marker='s', ls=':', label='Random ablation'),
}


# ── データ・モデル準備 ────────────────────────────────────────────────────────

def load_and_fit():
    """sparse activationsを読み込み、MaxAbsScaler + RidgeCVをtrainにfit。
    ablationはtestのスケール済み行列（scalerはtrainにのみfit）で行う。"""
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
    X_te_sc = scaler.transform(X[test_idx])
    y_tr = y[train_idx]
    y_te = y[test_idx]

    alphas = np.logspace(-1, 4, 30)
    ridge = RidgeCV(alphas=alphas, cv=CV_FOLDS, scoring='neg_root_mean_squared_error')
    ridge.fit(X_tr_sc, y_tr)
    print(f"Ridge α={ridge.alpha_:.4f}")

    return ridge, X_te_sc, y_te


def evaluate(y_true, y_pred):
    return {
        'rmse':     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':      float(mean_absolute_error(y_true, y_pred)),
        'r2':       float(r2_score(y_true, y_pred)),
        'pearson':  float(pearsonr(y_true, y_pred)[0]),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
    }


def bootstrap_ci_metrics(y_true, y_pred, n_boot=N_CI_BOOTSTRAP, seed=RANDOM_STATE):
    """percentile法95%CI。返り値: {metric: (lo, hi)}"""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    boot = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        m = evaluate(y_true[idx], y_pred[idx])
        for k in boot:
            boot[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}


def compute_test_ci(ridge, X_te_sc, y_te):
    """top-|w| ablation（test bootstrap）とrandom ablation（nested bootstrap）の
    95%CIを算出し、long形式のDataFrame（n, type, metric, point, ci_lo, ci_hi）を返す。"""
    weights = ridge.coef_
    rank_by_abs = np.argsort(np.abs(weights))[::-1]
    n_features = X_te_sc.shape[1]
    n_samples = X_te_sc.shape[0]

    rng = np.random.default_rng(RANDOM_STATE + 1)
    rows = []

    # n=0（ablationなし）: top-k-abs/randomとも同一のbaseline
    y_pred0 = ridge.predict(X_te_sc)
    point0 = evaluate(y_te, y_pred0)
    ci0 = bootstrap_ci_metrics(y_te, y_pred0)
    for m in METRICS:
        for t in ['top-k-abs', 'random_mean']:
            rows.append({'n': 0, 'type': t, 'metric': m,
                         'point': point0[m], 'ci_lo': ci0[m][0], 'ci_hi': ci0[m][1]})

    for n in ABLATION_NS:
        if n == 0:
            continue

        # ── top-|w|: 予測固定 + テストセットbootstrap ──
        X_abl = X_te_sc.copy()
        X_abl[:, rank_by_abs[:n]] = 0.0
        y_pred = ridge.predict(X_abl)
        point = evaluate(y_te, y_pred)
        ci = bootstrap_ci_metrics(y_te, y_pred)
        for m in METRICS:
            rows.append({'n': n, 'type': 'top-k-abs', 'metric': m,
                         'point': point[m], 'ci_lo': ci[m][0], 'ci_hi': ci[m][1]})

        # ── random: nested bootstrap（特徴選択 + テスト再抽出を同時に乱数化）──
        trial_point  = {m: [] for m in METRICS}
        trial_nested = {m: [] for m in METRICS}
        for _ in range(N_CI_RANDOM_TRIALS):
            rand_idx = rng.choice(n_features, size=n, replace=False)
            X_r = X_te_sc.copy()
            X_r[:, rand_idx] = 0.0
            y_pred_r = ridge.predict(X_r)

            m_orig = evaluate(y_te, y_pred_r)
            for m in METRICS:
                trial_point[m].append(m_orig[m])

            boot_idx = rng.integers(0, n_samples, size=n_samples)
            m_boot = evaluate(y_te[boot_idx], y_pred_r[boot_idx])
            for m in METRICS:
                trial_nested[m].append(m_boot[m])

        for m in METRICS:
            pt_arr = np.array(trial_point[m])
            ci_arr = np.array(trial_nested[m])
            rows.append({'n': n, 'type': 'random_mean', 'metric': m,
                         'point': float(pt_arr.mean()),
                         'ci_lo': float(np.percentile(ci_arr, 2.5)),
                         'ci_hi': float(np.percentile(ci_arr, 97.5))})

        print(f"  n={n:4d} done")

    return pd.DataFrame(rows)


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_fig_s4(df_ci, out_path):
    """Fig S4: (a) RMSE, (b) MAE, (c) R² を1行3パネルで表示（bootstrap 95%CI帯付き）"""
    fig, axes = plt.subplots(1, len(FIGURE_METRICS), figsize=(4.6 * len(FIGURE_METRICS), 3.8))

    for ax, metric, panel in zip(axes, FIGURE_METRICS, 'abc'):
        for abl_type, style in STYLE.items():
            sub = df_ci[(df_ci['type'] == abl_type) & (df_ci['metric'] == metric)].sort_values('n')
            ax.plot(sub['n'], sub['point'], marker=style['marker'], ls=style['ls'],
                    color=style['color'], linewidth=1.5, markersize=5,
                    label=style['label'], zorder=3)
            ax.fill_between(sub['n'], sub['ci_lo'], sub['ci_hi'],
                            color=style['color'], alpha=0.18, zorder=1)
        ax.set_title(f'({panel}) {METRICS_LABEL[metric]}', fontsize=11)
        ax.set_xlabel('Number of ablated features', fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    axes[0].legend(fontsize=9, loc='best')
    fig.suptitle('Feature ablation on test set (n=147) with 95% CI\n'
                 'Top-|w| ablation (test bootstrap) vs random ablation (nested bootstrap)', fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  図保存: {out_path}")


def main():
    (OUTPUT_DIR / 'figure').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)

    ridge, X_te_sc, y_te = load_and_fit()

    weights = ridge.coef_
    print(f"全{len(weights)}特徴中 |w|>0.1: {(np.abs(weights) > 0.1).sum()}個\n")

    print("=== テストセットのablation + bootstrap 95%CI計算 ===")
    df_ci = compute_test_ci(ridge, X_te_sc, y_te)
    df_ci.to_csv(OUTPUT_DIR / 'tables' / 'ablation_test_bootstrap_ci.csv', index=False)
    print(f"テーブル保存: {OUTPUT_DIR}/tables/ablation_test_bootstrap_ci.csv")

    plot_fig_s4(df_ci, OUTPUT_DIR / 'figure' / 'FigS4_feature_ablation.png')

    print(f"\n完了。出力先: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
