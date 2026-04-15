#!/usr/bin/env python3
"""
SFT Analysis Script
Converted from analyze_SFT_cleaned.ipynb

This script analyzes sparse activation features for protein thermal stability prediction,
generating an HTML report with all analysis results and visualizations.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables for storing results for HTML report
REPORT_DATA = {
    'sparsity_stats': {},
    'model_metrics': {},
    'figures': [],
    'tables': [],
    'feature_analyses': [],
    'detailed_features': {}  # Store detailed analysis for top/bottom features
}

# Normalized score at or below this value is treated as non-firing in score_tables.
DEFAULT_SCORE_FIRING_THRESHOLD = 0.05


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze SFT (Sparse Fine-Tuning) features for protein thermal stability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--data-dir', type=str,
                        default='/data3/taihei/matsunaga-repos/interPLM/interplm/nbthermo/nbthermo_embedding_esm_ssft-sft_8m',
                        help='Directory containing dense activations')
    parser.add_argument('--sparse-dir', type=str,
                        default='/data3/taihei/matsunaga-repos/interPLM/interplm/nbthermo/nbthermo_embedding_sparse_ssft-sft_8m',
                        help='Directory containing sparse activations')
    parser.add_argument('--tm-data', type=str,
                        default='/data3/taihei/matsunaga-repos/nanobody-thermostability-sae/data/nbbench/thermo-seq/vhh_thermo_seq.csv',
                        help='Path to TM dataset CSV')
    
    # Model parameters
    parser.add_argument('--layer', type=int, default=6,
                        help='Layer number for activations')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size for train/test split')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--cv-folds', type=int, default=10,
                        help='Number of CV folds for RidgeCV')
    
    # Feature analysis
    parser.add_argument('--feature-indices', type=int, nargs='+', default=[1688, 1677],
                        help='Feature indices to analyze in detail')
    parser.add_argument('--tm-extreme-fraction', type=float, default=0.2,
                        help='Top/bottom fraction of sequences by Tm for score-table AHO mean profile')
    parser.add_argument('--score-firing-threshold', type=float, default=DEFAULT_SCORE_FIRING_THRESHOLD,
                        help='Score positions at or below this value are omitted from score_tables (non-firing)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for results')
    
    return parser.parse_args()


def create_output_directories(output_dir):
    """Create necessary output directories"""
    dirs = {
        'main': output_dir,
        'figures': os.path.join(output_dir, 'figure'),
        'tables': os.path.join(output_dir, 'score_tables'),
        'data': os.path.join(output_dir, 'data')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def load_data(data_dir, sparse_dir, tm_data_path, layer):
    """Load TM data and activations"""
    print(f"Loading TM data from: {tm_data_path}")
    df = pd.read_csv(tm_data_path)
    
    y = df['label'].values
    print(f"データ数: {len(y)}, Tm範囲: {y.min():.1f} - {y.max():.1f}")
    
    print(f"Loading dense activations from: {data_dir}")
    data_dir = Path(data_dir)
    dense_proteins = torch.load(data_dir / f'all_sequences_layer_{layer}_activations.pt', map_location='cpu')
    
    print(f"Loading sparse activations from: {sparse_dir}")
    sparse_dir = Path(sparse_dir)
    sparse_proteins = torch.load(sparse_dir / f'all_sequences_layer_{layer}_sparse_activations.pt', map_location='cpu')
    
    print(f"Dense: {len(dense_proteins)}個のタンパク質, 次元={dense_proteins[0].shape[1]}")
    print(f"Sparse: {len(sparse_proteins)}個のタンパク質, 次元={sparse_proteins[0].shape[1]}")
    
    return df, y, dense_proteins, sparse_proteins


def mean_pool_proteins(protein_list):
    """Mean pool proteins at sequence level"""
    return torch.stack([protein.mean(dim=0) for protein in protein_list])


def analyze_sae_sparsity(sparse_before_pooling, output_dir):
    """Analyze SAE sparsity in detail"""
    print("=== SAE Sparsity Analysis ===")
    
    all_sparse_activations = torch.cat(sparse_before_pooling, dim=0)
    total_elements = all_sparse_activations.numel()
    nonzero_elements = torch.count_nonzero(all_sparse_activations).item()
    overall_sparsity = 1 - (nonzero_elements / total_elements)
    
    print(f"Total elements: {total_elements:,}")
    print(f"Non-zero elements: {nonzero_elements:,}")
    print(f"Overall sparsity: {overall_sparsity:.4f}")
    print(f"Active ratio: {1-overall_sparsity:.4f}")
    
    # Feature-wise sparsity
    feature_sparsity = []
    for feature_idx in range(all_sparse_activations.shape[1]):
        feature_activations = all_sparse_activations[:, feature_idx]
        feature_nonzero = torch.count_nonzero(feature_activations).item()
        feature_sparsity_val = 1 - (feature_nonzero / len(feature_activations))
        feature_sparsity.append(feature_sparsity_val)
    
    feature_sparsity = np.array(feature_sparsity)
    
    print(f"\nFeature-wise sparsity statistics:")
    print(f"Mean feature sparsity: {feature_sparsity.mean():.4f}")
    print(f"Std feature sparsity: {feature_sparsity.std():.4f}")
    print(f"Min feature sparsity: {feature_sparsity.min():.4f}")
    print(f"Max feature sparsity: {feature_sparsity.max():.4f}")
    
    firing_rates = 1 - feature_sparsity
    
    print(f"\nFeature firing rate distribution:")
    print(f"Features firing >50% of time: {np.sum(firing_rates > 0.5)}")
    print(f"Features firing >20% of time: {np.sum(firing_rates > 0.2)}")
    print(f"Features firing >10% of time: {np.sum(firing_rates > 0.1)}")
    print(f"Features firing >5% of time: {np.sum(firing_rates > 0.05)}")
    print(f"Features firing >1% of time: {np.sum(firing_rates > 0.01)}")
    print(f"Dead features (never firing): {np.sum(firing_rates == 0)}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(firing_rates, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Feature Firing Rate')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Distribution of Feature Firing Rates')
    ax1.axvline(firing_rates.mean(), color='red', linestyle='--', 
                label=f'Mean: {firing_rates.mean():.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    sorted_rates = np.sort(firing_rates)[::-1]
    ax2.plot(range(len(sorted_rates)), sorted_rates)
    ax2.set_xlabel('Feature Rank (sorted by firing rate)')
    ax2.set_ylabel('Firing Rate')
    ax2.set_title('Feature Firing Rates (Ranked)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figure', 'sae_sparsity_analysis.png')
    plt.savefig(fig_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    # Store for report
    REPORT_DATA['sparsity_stats'] = {
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'overall_sparsity': overall_sparsity,
        'active_ratio': 1 - overall_sparsity,
        'mean_feature_sparsity': feature_sparsity.mean(),
        'std_feature_sparsity': feature_sparsity.std(),
        'firing_rate_distribution': {
            'gt_50pct': int(np.sum(firing_rates > 0.5)),
            'gt_20pct': int(np.sum(firing_rates > 0.2)),
            'gt_10pct': int(np.sum(firing_rates > 0.1)),
            'gt_5pct': int(np.sum(firing_rates > 0.05)),
            'gt_1pct': int(np.sum(firing_rates > 0.01)),
            'dead': int(np.sum(firing_rates == 0))
        }
    }
    REPORT_DATA['figures'].append({
        'path': 'figure/sae_sparsity_analysis.png',
        'title': 'SAE Sparsity Analysis',
        'caption': 'Distribution and ranking of feature firing rates'
    })
    
    return firing_rates, feature_sparsity


def train_and_evaluate_model(X_dense, X_sparse, y, df, tm_data_path, cv_folds, output_dir,
                             test_size=0.2, random_state=42):
    """Train RidgeCV models and evaluate performance (nbbench train/test split when available)."""
    print("=== Training Models ===")

    thermo_dir = Path(tm_data_path).parent
    train_csv = thermo_dir / "train.csv"
    test_csv = thermo_dir / "test.csv"

    if train_csv.is_file() and test_csv.is_file():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        seq_to_idx = {seq: i for i, seq in enumerate(df["seq"])}
        train_idx = np.array([seq_to_idx[s] for s in train_df["seq"]], dtype=int)
        test_idx = np.array([seq_to_idx[s] for s in test_df["seq"]], dtype=int)
        X_dense_train = X_dense[train_idx]
        X_dense_test = X_dense[test_idx]
        X_sparse_train = X_sparse[train_idx]
        X_sparse_test = X_sparse[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        print(f"Train n={len(train_idx)}, Test n={len(test_idx)} (nbbench split, val excluded)")
    else:
        print(
            f"Warning: {train_csv.name} / {test_csv.name} not found under {thermo_dir}; "
            "using random train_test_split."
        )
        X_dense_train, X_dense_test, y_train, y_test = train_test_split(
            X_dense, y, test_size=test_size, random_state=random_state
        )
        X_sparse_train, X_sparse_test, _, _ = train_test_split(
            X_sparse, y, test_size=test_size, random_state=random_state
        )

    scaler_dense = MaxAbsScaler()
    scaler_sparse = MaxAbsScaler()

    X_dense_train_scaled = scaler_dense.fit_transform(X_dense_train)
    X_dense_test_scaled = scaler_dense.transform(X_dense_test)
    X_sparse_train_scaled = scaler_sparse.fit_transform(X_sparse_train)
    X_sparse_test_scaled = scaler_sparse.transform(X_sparse_test)

    alphas = np.logspace(-1, 4, 30)

    ridgecv_dense = RidgeCV(alphas=alphas, cv=cv_folds, scoring="neg_root_mean_squared_error")
    ridgecv_sparse = RidgeCV(alphas=alphas, cv=cv_folds, scoring="neg_root_mean_squared_error")

    ridgecv_dense.fit(X_dense_train_scaled, y_train)
    ridgecv_sparse.fit(X_sparse_train_scaled, y_train)

    print(f"Dense: optimal α={ridgecv_dense.alpha_}")
    print(f"Sparse: optimal α={ridgecv_sparse.alpha_}")

    y_pred_dense = ridgecv_dense.predict(X_dense_test_scaled)
    y_pred_sparse = ridgecv_sparse.predict(X_sparse_test_scaled)

    def evaluate_model(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        print(
            f"{name}: RMSE={rmse:.3f}, R²={r2:.3f}, MAE={mae:.3f}, "
            f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
        )
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman,
        }

    print("=== Performance evaluation ===")
    dense_metrics = evaluate_model(y_test, y_pred_dense, "Dense")
    sparse_metrics = evaluate_model(y_test, y_pred_sparse, "Sparse")
    
    # Store metrics
    REPORT_DATA['model_metrics'] = {
        'dense': dense_metrics,
        'sparse': sparse_metrics,
        'dense_alpha': float(ridgecv_dense.alpha_),
        'sparse_alpha': float(ridgecv_sparse.alpha_)
    }
    
    return {
        'ridgecv_dense': ridgecv_dense,
        'ridgecv_sparse': ridgecv_sparse,
        'y_test': y_test,
        'y_pred_dense': y_pred_dense,
        'y_pred_sparse': y_pred_sparse,
        'scaler_dense': scaler_dense,
        'scaler_sparse': scaler_sparse
    }


def create_joint_plot(tm_values, predictions, metrics,
                      title=None,
                      title_size=8,
                      figsize=(2, 2),
                      scatter_size=10,
                      tick_size=5,
                      label_size=6,
                      stats_size=6,
                      linespacing=1.5):
    """Create joint plot for prediction results (matches analyze_SAE_feature.ipynb)."""
    rmse = metrics['rmse']
    r2 = metrics['r2']
    mae = metrics['mae']
    spearman = metrics['spearman']

    sns.set_style("whitegrid", {"grid.color": "0.9"})
    BORDER_COLOR = 'black'
    line_width = 0.3

    jp = sns.jointplot(
        x=tm_values,
        y=predictions,
        kind="scatter",
        height=figsize[0],
        color="steelblue",
        edgecolor="w",
        s=scatter_size,
    )

    jp.ax_marg_x.set_position([0.05, 0.82, 0.7, 0.1])
    jp.ax_marg_y.set_position([0.82, 0.05, 0.1, 0.7])
    jp.ax_joint.set_position([0.05, 0.05, 0.7, 0.7])

    plot_min, plot_max = 25, 99
    jp.ax_joint.set_xlim(plot_min, plot_max)
    jp.ax_joint.set_ylim(plot_min, plot_max)

    for spine in jp.ax_joint.spines.values():
        if spine in [jp.ax_joint.spines['top'], jp.ax_joint.spines['right']]:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(line_width)

    for spine in jp.ax_marg_x.spines.values():
        if spine in [jp.ax_marg_x.spines['top'], jp.ax_marg_x.spines['left'], jp.ax_marg_x.spines['right']]:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(line_width)
    jp.ax_marg_x.grid(False)
    jp.ax_marg_x.tick_params(axis='x', direction="out", bottom=True, length=2, width=line_width)

    for spine in jp.ax_marg_y.spines.values():
        if spine in [jp.ax_marg_y.spines['top'], jp.ax_marg_y.spines['bottom'], jp.ax_marg_y.spines['right']]:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(line_width)
    jp.ax_marg_y.grid(False)
    jp.ax_marg_y.tick_params(axis='y', direction="out", left=True, length=2, width=line_width)

    jp.ax_joint.set_xticks([30, 40, 50, 60, 70, 80, 90])
    jp.ax_joint.tick_params(
        axis='both',
        direction="out",
        bottom=True,
        left=True,
        length=3,
        width=line_width,
        labelsize=tick_size,
    )

    jp.ax_joint.grid(False)
    jp.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=0.3)

    jp.set_axis_labels("True Tm (°C)", "Predicted Tm (°C)", fontsize=label_size)

    if title:
        jp.fig.suptitle(title, fontsize=title_size)

    stats_text = (
        f"Spearman: {spearman:.3f}\n"
        f"R²: {r2:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"MAE: {mae:.3f}"
    )

    jp.ax_joint.text(
        0.05,
        0.95,
        stats_text,
        transform=jp.ax_joint.transAxes,
        fontsize=stats_size,
        verticalalignment="top",
        linespacing=linespacing,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    return jp


def save_joint_plot(jp, output_dir, filename, dpi=350):
    """Save a joint plot to output_dir/figure/ (matches notebook save_plot pattern)."""
    fig_dir = os.path.join(output_dir, 'figure')
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, filename)
    jp.savefig(save_path, dpi=dpi, format="png", bbox_inches='tight')
    plt.close()
    return save_path


def create_correlation_plot(y_pred_dense, y_pred_sparse, output_dir, figsize=(2, 2)):
    """Create correlation plot between dense and sparse predictions"""
    pearson_corr, pearson_p = pearsonr(y_pred_dense, y_pred_sparse)
    r2 = r2_score(y_pred_dense, y_pred_sparse)
    
    sns.set_style("whitegrid", {"grid.color": "0.9"})
    BORDER_COLOR = '0.5'
    line_width = 0.5
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.scatter(y_pred_dense, y_pred_sparse, 
               color="steelblue", 
               alpha=0.6, 
               s=8,
               linewidth=0.5)
    
    plot_min, plot_max = 40, 99
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=0.3)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(line_width)
    
    ax.set_xlabel("Predicted Tm from SFT emb (°C)", fontsize=5)
    ax.set_ylabel("Predicted Tm from Sparse rep (°C)", fontsize=5)
    ax.grid(False)
    
    stats_text = f"Pearson: {pearson_corr:.3f}\nR²: {r2:.3f}"
    
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round",
                    facecolor="white",
                    alpha=0.5))
    
    ax.tick_params(axis='both', width=line_width, labelsize=5)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure', 'original_vs_sparse_correlation_SFT.png')
    plt.savefig(save_path, dpi=350, format="png", bbox_inches='tight')
    plt.close()
    
    REPORT_DATA['figures'].append({
        'path': 'figure/original_vs_sparse_correlation_SFT.png',
        'title': 'Correlation between Dense and Sparse Predictions',
        'caption': f'Pearson correlation: {pearson_corr:.3f}, R²: {r2:.3f}'
    })
    
    return save_path


def analyze_feature_importance(model_results, output_dir):
    """Analyze feature importance from sparse model"""
    print("=== Feature Importance Analysis ===")
    
    sparse_weights = model_results['ridgecv_sparse'].coef_
    print(f"Sparse重みの形状: {sparse_weights.shape}")
    
    max_weight_idx = np.argmax(np.abs(sparse_weights))
    max_weight_value = sparse_weights[max_weight_idx]
    
    print(f"最重要特徴量インデックス: {max_weight_idx}")
    print(f"最重要特徴量の重み: {max_weight_value:.6f}")
    
    top_indices = np.argsort(np.abs(sparse_weights))[-10:][::-1]
    print(f"\n上位10個の重要特徴量:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. インデックス={idx}, 重み={sparse_weights[idx]:.6f}")
    
    # Visualization
    indices_sorted = np.argsort(-np.abs(sparse_weights))
    sorted_weights = np.abs(sparse_weights[indices_sorted])
    ranks = np.arange(len(sorted_weights))
    
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, sorted_weights, marker='', linestyle='-', linewidth=1)
    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Weight", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure', 'ridge_weights_ranked.png')
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    REPORT_DATA['figures'].append({
        'path': 'figure/ridge_weights_ranked.png',
        'title': 'Ridge Regression Weights (Ranked)',
        'caption': 'Feature importance ranked by absolute weight value'
    })
    
    return sparse_weights


def analyze_specific_features(sparse_before_pooling, df, feature_indices, output_dir):
    """Analyze specific features in detail"""
    print(f"=== Analyzing Features: {feature_indices} ===")
    
    # Correlations for all features
    correlations = {}
    for feat_idx in feature_indices:
        activations_per_protein = []
        for i, protein_tensor in enumerate(sparse_before_pooling):
            feature_activations = protein_tensor[:, feat_idx].numpy()
            activations_per_protein.append(feature_activations)
        
        protein_feature_strengths = []
        for i, activations in enumerate(activations_per_protein):
            mean_activation = np.mean(activations)
            protein_feature_strengths.append({
                'protein_id': i,
                'mean_activation': mean_activation,
                'label': df.iloc[i]['label']
            })
        
        mean_activations = [stats['mean_activation'] for stats in protein_feature_strengths]
        tm_values = [stats['label'] for stats in protein_feature_strengths]
        correlation = np.corrcoef(mean_activations, tm_values)[0, 1]
        
        correlations[feat_idx] = correlation
        print(f"Feature {feat_idx}: correlation with Tm = {correlation:.4f}")
    
    # 相関図の描画は行わない（削除）
    
    REPORT_DATA['feature_analyses'] = correlations
    
    return correlations


def normalize_activations(sparse_before_pooling, df, feature_idx):
    """Normalize activations to 0-1 range"""
    activations_per_protein = []
    for i, protein_tensor in enumerate(sparse_before_pooling):
        feature_activations = protein_tensor[:, feature_idx].numpy()
        activations_per_protein.append(feature_activations)
    
    protein_feature_strengths = []
    mean_activations = []
    
    for i, activations in enumerate(activations_per_protein):
        mean_activation = np.mean(activations)
        protein_feature_strengths.append({
            'protein_id': i,
            'mean_activation': mean_activation,
            'label': df.iloc[i]['label']
        })
        mean_activations.append(mean_activation)
    
    mean_activations = np.array(mean_activations)
    min_activation = np.min(mean_activations)
    max_activation = np.max(mean_activations)
    
    if max_activation != min_activation:
        normalized_activations = (mean_activations - min_activation) / (max_activation - min_activation)
        for i, protein_data in enumerate(protein_feature_strengths):
            protein_data['mean_activation'] = normalized_activations[i]
    
    return protein_feature_strengths


def get_top_bottom_features(sparse_weights, n=10):
    """Extract top n positive and bottom n negative features by weight"""
    pos_idx = np.argsort(sparse_weights)[-n:][::-1]  # Top n positive
    neg_idx = np.argsort(sparse_weights)[:n][::-1]   # Bottom n negative (most negative first)
    return pos_idx, neg_idx


def build_ungapped_to_aho_map(seq_aho):
    """
    Build mapping from ungapped sequence indices to AHO positions
    """
    mapping = []
    for i, ch in enumerate(seq_aho):
        if ch != '-':
            mapping.append(i + 1)  # AHO positions are 1-based
    return mapping


def build_aho_to_ungapped_resi(seq_aho):
    """
    For one gapped AHO string, map AHO column (1-based) to ungapped residue index (1-based).
    Columns where this sequence has a gap are absent from the returned dict.
    """
    ungapped_to_aho = build_ungapped_to_aho_map(seq_aho)
    aho_to_resi = {}
    for u_idx, aho_1based in enumerate(ungapped_to_aho):
        aho_to_resi[int(aho_1based)] = u_idx + 1
    return aho_to_resi


def build_aho_activation_matrix(sparse_before_pooling, df, feature_idx, aho_len=149):
    """
    Stack AHO-aligned activations for one SAE feature across all proteins.
    Row i matches df.iloc[i] / sparse_before_pooling[i].

    Returns:
        M: (N, aho_len) float array
        meta: list of dicts with keys 'id' (row index), 'label' (Tm)
    """
    num_proteins = len(sparse_before_pooling)
    rows = []
    meta = []
    for i in range(num_proteins):
        protein_tensor = sparse_before_pooling[i]
        activations = protein_tensor[:, feature_idx].numpy()
        tm_val = df.iloc[i]['label']
        aho_gapped_sequence = df.iloc[i]['sequence_aho']
        ungapped_to_aho = build_ungapped_to_aho_map(aho_gapped_sequence)
        aho_aligned_activations = np.zeros(aho_len)
        L = min(len(activations), len(ungapped_to_aho))
        for u_idx in range(L):
            aho_pos = ungapped_to_aho[u_idx] - 1
            if aho_pos < aho_len:
                aho_aligned_activations[aho_pos] = activations[u_idx]
        rows.append(aho_aligned_activations)
        meta.append({'id': i, 'label': tm_val})
    M = np.stack(rows, axis=0)
    return M, meta


def create_tm_correlation_plot(protein_feature_strengths, feature_idx, output_dir):
    """Create correlation plot between mean activation and Tm"""
    mean_activations = [p['mean_activation'] for p in protein_feature_strengths]
    tm_values = [p['label'] for p in protein_feature_strengths]
    
    correlation = np.corrcoef(mean_activations, tm_values)[0, 1]
    
    fig, ax = plt.subplots(figsize=(3, 3))
    
    ax.scatter(mean_activations, tm_values, alpha=0.6, s=15, color='steelblue')
    ax.set_xlabel('Mean Activation Strength', fontsize=10)
    ax.set_ylabel('Tm (°C)', fontsize=10)
    ax.set_title(f'Feature {feature_idx} vs Tm\n(r={correlation:.3f})', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure', f'feature_{feature_idx}_tm_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path, correlation


def create_aho_heatmaps(sparse_before_pooling, df, feature_idx, output_dir,
                        is_positive_weight=True, aho_len=149):
    """
    Create AHO-aligned heatmap for all proteins sorted by Tm (high to low)
    using a seaborn heatmap to keep figure height manageable.
    Returns both the heatmap path and the top 5 mean-activation AHO positions.
    """
    M, meta = build_aho_activation_matrix(
        sparse_before_pooling, df, feature_idx, aho_len=aho_len
    )
    order = np.argsort([-m['label'] for m in meta])
    all_protein_data = [
        {
            'id': meta[j]['id'],
            'label': meta[j]['label'],
            'aho_aligned_activations': M[j],
        }
        for j in order
    ]
    
    # Create single heatmap with all sequences
    M = np.array([p['aho_aligned_activations'] for p in all_protein_data])
    mean_activation_per_pos = M.mean(axis=0)
    
    # Identify top 5 AHO positions by mean activation
    top_pos_indices = np.argsort(mean_activation_per_pos)[-5:][::-1]
    top_positions = [
        {
            'position': int(idx + 1),
            'mean_activation': float(mean_activation_per_pos[idx])
        }
        for idx in top_pos_indices
    ]
    
    # Build DataFrame as in reference notebook for seaborn heatmap
    heatmap_df = pd.DataFrame(
        M,
        columns=[f"AHO{j+1}" for j in range(aho_len)],
        index=[f"ID {p['id']} (Tm={p['label']:.1f}°C)" for p in all_protein_data]
    )
    
    # Figure size fixed to keep output manageable
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = 'Reds' if is_positive_weight else 'Blues'
    cbar = sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap=cmap,
        cbar_kws={'label': 'Activation'},
        yticklabels=False,
        xticklabels=False
    )
    
    # Adjust colorbar font size
    cbar.figure.axes[-1].tick_params(labelsize=8)
    cbar.figure.axes[-1].set_ylabel('Activation', fontsize=8)
    
    # Configure ticks similar to notebook example (subset for readability)
    x_ticks = list(range(0, aho_len, 10))
    x_labels = [f"{i+1}" for i in x_ticks]
    ax.set_xticks([x + 0.5 for x in x_ticks])
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    y_ticks = list(range(0, len(all_protein_data), 50))
    y_labels = [f"rank {i+1}" for i in y_ticks]
    ax.set_yticks([y + 0.5 for y in y_ticks])
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlabel("AHO aligned position", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sequences (ranked by Tm)", fontsize=12, fontweight='bold')
    #ax.set_title(
    #    f'Feature {feature_idx} - All Sequences Sorted by Tm (High to Low)',
    #    fontsize=10
    #)
    
    plt.tight_layout()
    
    filename = f'aho_heatmap_all_{feature_idx}.png'
    save_path = os.path.join(output_dir, 'figure', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path, top_positions


def extract_firing_positions(
    sparse_before_pooling,
    df,
    feature_idx,
    output_dir,
    is_positive_weight=True,
    tm_extreme_fraction=0.2,
    aho_len=149,
    score_firing_threshold=DEFAULT_SCORE_FIRING_THRESHOLD,
):
    """
    Aggregate firing positions over the top or bottom Tm fraction of sequences.

    Mean activation is computed per AHO position across the selected group, then
    scores match the previous convention: ReLU on the mean profile, divided by its max.

    ``resi`` follows the legacy convention: ungapped residue index (1-based) for the single
    highest-Tm sequence (positive weight) or lowest-Tm sequence (negative weight), at each
    ``aho_pos``. Where that reference sequence has a gap, ``resi`` is empty in the CSV.

    Args:
        is_positive_weight: If True, use highest-Tm fraction; if False, lowest-Tm fraction.
        tm_extreme_fraction: Fraction of sequences in each tail (e.g. 0.2 for top/bottom 20%).
        score_firing_threshold: Positions with score <= this value are dropped from the CSV.
    """
    M, meta = build_aho_activation_matrix(
        sparse_before_pooling, df, feature_idx, aho_len=aho_len
    )
    tm_values = np.array([m['label'] for m in meta], dtype=float)
    N = len(tm_values)
    k = max(1, int(np.ceil(tm_extreme_fraction * N)))
    pct_int = int(round(tm_extreme_fraction * 100))

    if is_positive_weight:
        group_idx = np.argsort(-tm_values)[:k]
        label = f'top{pct_int}pct'
        tm_side = 'high'
        filename_suffix = f'top{pct_int}pct_hightm'
    else:
        group_idx = np.argsort(tm_values)[:k]
        label = f'bottom{pct_int}pct'
        tm_side = 'low'
        filename_suffix = f'bottom{pct_int}pct_lowtm'

    mean_activation = M[group_idx].mean(axis=0)
    acts_pos = np.clip(mean_activation, a_min=0.0, a_max=None)
    max_pos = acts_pos.max() if acts_pos.size > 0 else 0.0
    if max_pos > 0:
        scores = acts_pos / max_pos
    else:
        scores = np.zeros_like(acts_pos)

    ref_idx = int(np.argmax(tm_values) if is_positive_weight else np.argmin(tm_values))
    ref_seq_aho = df.iloc[ref_idx]['sequence_aho']
    aho_to_resi = build_aho_to_ungapped_resi(ref_seq_aho)

    pos_1based = np.arange(1, aho_len + 1, dtype=int)
    resi_vals = [
        aho_to_resi[int(ap)] if int(ap) in aho_to_resi else pd.NA
        for ap in pos_1based
    ]
    df_scores = pd.DataFrame({
        'resi': pd.array(resi_vals, dtype=pd.Int64Dtype()),
        'aho_pos': pos_1based,
        'mean_activation': mean_activation.astype(float),
        'score': scores.astype(float),
        'raw_activation': mean_activation.astype(float),
    })
    df_scores = df_scores[df_scores['score'] > score_firing_threshold].reset_index(drop=True)

    filename = f'scores_feature{feature_idx}_{filename_suffix}.csv'
    out_csv = os.path.join(output_dir, 'score_tables', filename)
    df_scores.to_csv(out_csv, index=False)

    group_tms = tm_values[group_idx]
    return [{
        'label': label,
        'tm_side': tm_side,
        'tm_min': float(group_tms.min()),
        'tm_max': float(group_tms.max()),
        'tm_mean': float(group_tms.mean()),
        'n_sequences': int(k),
        'seq_ids': group_idx.astype(int).tolist(),
        'csv_path': filename,
        'num_firing': len(df_scores),
    }]


def analyze_feature_detailed(
    feature_idx,
    sparse_before_pooling,
    df,
    output_dir,
    is_positive_weight=True,
    tm_extreme_fraction=0.2,
    score_firing_threshold=DEFAULT_SCORE_FIRING_THRESHOLD,
):
    """
    Perform detailed analysis for a single feature:
    1. Correlation plot with Tm
    2. AHO-aligned heatmap (all sequences sorted by Tm)
    3. Score tables from mean AHO activations over top/bottom Tm fraction (by weight sign)
    
    Args:
        is_positive_weight: If True, extract only high Tm firing positions
                           If False, extract only low Tm firing positions
        tm_extreme_fraction: Tail fraction for score-table aggregation (e.g. 0.2).
        score_firing_threshold: Minimum normalized score to count as firing in score_tables.
    """
    print(f"\n=== Analyzing Feature {feature_idx} ===")
    
    # 1. Get activation strengths
    activations_per_protein = []
    for i, protein_tensor in enumerate(sparse_before_pooling):
        feature_activations = protein_tensor[:, feature_idx].numpy()
        activations_per_protein.append(feature_activations)
    
    protein_feature_strengths = []
    for i, activations in enumerate(activations_per_protein):
        mean_activation = np.mean(activations)
        protein_feature_strengths.append({
            'protein_id': i,
            'mean_activation': mean_activation,
            'label': df.iloc[i]['label']
        })
    
    # 2. Create correlation plot
    corr_path, correlation = create_tm_correlation_plot(
        protein_feature_strengths, feature_idx, output_dir
    )
    print(f"  Correlation with Tm: {correlation:.4f}")
    
    # 3. Create AHO heatmap (all sequences) and compute top AHO positions
    heatmap_path, top_positions = create_aho_heatmaps(
        sparse_before_pooling, df, feature_idx, output_dir,
        is_positive_weight=is_positive_weight
    )
    print(f"  Created heatmap for all sequences sorted by Tm")
    print(f"  Top AHO positions (mean activation): {top_positions}")
    
    # 4. Score table: mean AHO profile over top/bottom Tm fraction
    firing_results = extract_firing_positions(
        sparse_before_pooling,
        df,
        feature_idx,
        output_dir,
        is_positive_weight=is_positive_weight,
        tm_extreme_fraction=tm_extreme_fraction,
        score_firing_threshold=score_firing_threshold,
    )
    print(
        f"  Wrote score table for {firing_results[0]['n_sequences']} sequences "
        f"({'high' if is_positive_weight else 'low'} Tm tail, fraction={tm_extreme_fraction}, "
        f"score>{score_firing_threshold})"
    )
    
    return {
        'feature_idx': feature_idx,
        'correlation': correlation,
        'corr_plot': f'figure/feature_{feature_idx}_tm_correlation.png',
        'heatmap': f'figure/aho_heatmap_all_{feature_idx}.png',
        'firing_positions': firing_results,
        'top_positions': top_positions
    }


def generate_html_report(output_dir):
    """Generate comprehensive HTML report"""
    print("Generating HTML report...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFT Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .figure-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure-title {{
            font-weight: bold;
            margin: 10px 0;
        }}
        .figure-caption {{
            color: #666;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .metric-box {{
            display: inline-block;
            padding: 15px 20px;
            margin: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>SFT Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="container">
        <h2>Executive Summary</h2>
"""
    
    # Add sparsity summary
    if 'sparsity_stats' in REPORT_DATA:
        stats = REPORT_DATA['sparsity_stats']
        html_content += f"""
        <h3>SAE Sparsity</h3>
        <div class="metric-box">
            <div class="metric-value">{stats['overall_sparsity']:.4f}</div>
            <div class="metric-label">Overall Sparsity</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{stats['active_ratio']:.4f}</div>
            <div class="metric-label">Active Ratio</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{stats['mean_feature_sparsity']:.4f}</div>
            <div class="metric-label">Mean Feature Sparsity</div>
        </div>
"""
    
    # Add model performance summary
    if 'model_metrics' in REPORT_DATA:
        metrics = REPORT_DATA['model_metrics']
        html_content += f"""
        <h3>Model Performance</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Pearson</th>
                <th>Spearman</th>
                <th>Optimal α</th>
            </tr>
            <tr>
                <td>Dense</td>
                <td>{metrics['dense']['r2']:.3f}</td>
                <td>{metrics['dense']['rmse']:.3f}</td>
                <td>{metrics['dense']['mae']:.3f}</td>
                <td>{metrics['dense']['pearson']:.3f}</td>
                <td>{metrics['dense']['spearman']:.3f}</td>
                <td>{metrics['dense_alpha']:.3f}</td>
            </tr>
            <tr>
                <td>Sparse</td>
                <td>{metrics['sparse']['r2']:.3f}</td>
                <td>{metrics['sparse']['rmse']:.3f}</td>
                <td>{metrics['sparse']['mae']:.3f}</td>
                <td>{metrics['sparse']['pearson']:.3f}</td>
                <td>{metrics['sparse']['spearman']:.3f}</td>
                <td>{metrics['sparse_alpha']:.3f}</td>
            </tr>
        </table>
"""
    
    # Add sparsity details
    if 'sparsity_stats' in REPORT_DATA:
        stats = REPORT_DATA['sparsity_stats']
        html_content += f"""
    </div>
    
    <div class="container">
        <h2>SAE Sparsity Analysis</h2>
        <p><strong>Total Elements:</strong> {stats['total_elements']:,}</p>
        <p><strong>Non-zero Elements:</strong> {stats['nonzero_elements']:,}</p>
        
        <h3>Feature Firing Rate Distribution</h3>
        <table>
            <tr>
                <th>Criteria</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>Features firing >50% of time</td>
                <td>{stats['firing_rate_distribution']['gt_50pct']}</td>
            </tr>
            <tr>
                <td>Features firing >20% of time</td>
                <td>{stats['firing_rate_distribution']['gt_20pct']}</td>
            </tr>
            <tr>
                <td>Features firing >10% of time</td>
                <td>{stats['firing_rate_distribution']['gt_10pct']}</td>
            </tr>
            <tr>
                <td>Features firing >5% of time</td>
                <td>{stats['firing_rate_distribution']['gt_5pct']}</td>
            </tr>
            <tr>
                <td>Features firing >1% of time</td>
                <td>{stats['firing_rate_distribution']['gt_1pct']}</td>
            </tr>
            <tr>
                <td>Dead features (never firing)</td>
                <td>{stats['firing_rate_distribution']['dead']}</td>
            </tr>
        </table>
"""
    
    # Add figures
    for fig in REPORT_DATA['figures']:
        html_content += f"""
        <div class="figure-container">
            <div class="figure-title">{fig['title']}</div>
            <img src="{fig['path']}" alt="{fig['title']}">
            <div class="figure-caption">{fig['caption']}</div>
        </div>
"""
    
    # Add feature analysis
    if 'feature_analyses' in REPORT_DATA and REPORT_DATA['feature_analyses']:
        html_content += """
    </div>
    
    <div class="container">
        <h2>Feature Analysis</h2>
"""
        for feat_idx, correlation in REPORT_DATA['feature_analyses'].items():
            html_content += f"""
        <p><strong>Feature {feat_idx}:</strong> Correlation with Tm = {correlation:.4f}</p>
"""
    
    # Add detailed feature analysis (top/bottom features)
    if 'detailed_features' in REPORT_DATA and REPORT_DATA['detailed_features']:
        html_content += """
    </div>
    
    <div class="container">
        <h2>Detailed Feature Analysis (Top/Bottom 10 by Weight)</h2>
        <p>Analysis of the top 10 positive and bottom 10 negative features by Ridge regression weight.</p>
"""
        
        for feat_type in ['positive', 'negative']:
            if feat_type in REPORT_DATA['detailed_features']:
                html_content += f"""
        <h3>{feat_type.capitalize()} Weight Features</h3>
"""
                features = REPORT_DATA['detailed_features'][feat_type]
                
                for feat_data in features:
                    feat_idx = feat_data['feature_idx']
                    correlation = feat_data['correlation']
                    
                    html_content += f"""
        <div style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px;">
            <h4>Feature {feat_idx} (correlation: {correlation:.4f})</h4>
            
            <div class="figure-container">
                <div class="figure-title">Correlation with Tm</div>
                <img src="{feat_data['corr_plot']}" alt="Feature {feat_idx} Correlation">
            </div>
            
            <div class="figure-container">
                <div class="figure-title">All Sequences Sorted by Tm (High to Low) - AHO Aligned</div>
                <img src="{feat_data['heatmap']}" alt="Feature {feat_idx} Heatmap">
            </div>
            
            <h5>Top 5 AHO Positions by Mean Activation</h5>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>AHO Position</th>
                    <th>Mean Activation</th>
                </tr>
"""
                    for rank, pos_data in enumerate(feat_data['top_positions'], start=1):
                        html_content += f"""
                <tr>
                    <td>{rank}</td>
                    <td>{pos_data['position']}</td>
                    <td>{pos_data['mean_activation']:.4f}</td>
                </tr>
"""
                    html_content += """
            </table>
            
            <h5>Score tables (mean AHO profile, Tm tail)</h5>
            <table>
                <tr>
                    <th>Group</th>
                    <th>Tm range (°C)</th>
                    <th>N sequences</th>
                    <th>Firing AHO positions</th>
                    <th>CSV File</th>
                </tr>
"""
                    for fp in feat_data['firing_positions']:
                        html_content += f"""
                <tr>
                    <td>{fp['label']} ({fp['tm_side']} Tm)</td>
                    <td>{fp['tm_min']:.1f}–{fp['tm_max']:.1f} (mean {fp['tm_mean']:.1f})</td>
                    <td>{fp['n_sequences']}</td>
                    <td>{fp['num_firing']}</td>
                    <td><a href="score_tables/{fp['csv_path']}">{fp['csv_path']}</a></td>
                </tr>
"""
                    
                    html_content += """
            </table>
        </div>
"""
    
    html_content += """
    </div>
    
    <footer>
        <p style="text-align: center; color: #7f8c8d;">Report generated by analyze_sft.py</p>
    </footer>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, 'report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {html_path}")
    return html_path


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Create output directories
    dirs = create_output_directories(args.output_dir)
    
    # Load data
    df, y, dense_proteins, sparse_proteins = load_data(
        args.data_dir, args.sparse_dir, args.tm_data, args.layer
    )
    
    # Mean pooling
    sparse_before_pooling = sparse_proteins
    X_dense = mean_pool_proteins(dense_proteins).numpy()
    X_sparse = mean_pool_proteins(sparse_proteins).numpy()
    
    print(f"Dense特徴量: {X_dense.shape}")
    print(f"Sparse特徴量: {X_sparse.shape}")
    
    # SAE sparsity analysis
    firing_rates, feature_sparsity = analyze_sae_sparsity(sparse_before_pooling, args.output_dir)
    
    # Train and evaluate models
    model_results = train_and_evaluate_model(
        X_dense, X_sparse, y, df, args.tm_data, args.cv_folds, args.output_dir,
        test_size=args.test_size, random_state=args.random_state,
    )

    jp_sparse = create_joint_plot(
        model_results['y_test'],
        model_results['y_pred_sparse'],
        REPORT_DATA['model_metrics']['sparse'],
        figsize=(1.8, 1.8),
        scatter_size=10,
        tick_size=5,
        label_size=5,
        stats_size=5,
    )
    joint_path = save_joint_plot(jp_sparse, args.output_dir, 'evaluation_tm_Sparse_SAE_SFT.png')
    REPORT_DATA['figures'].append({
        'path': 'figure/evaluation_tm_Sparse_SAE_SFT.png',
        'title': 'Sparse SAE Model Performance',
        'caption': f"RMSE: {REPORT_DATA['model_metrics']['sparse']['rmse']:.3f}, R²: {REPORT_DATA['model_metrics']['sparse']['r2']:.3f}"
    })
    
    # Create correlation plot
    create_correlation_plot(
        model_results['y_pred_sparse'], model_results['y_pred_dense'],
        args.output_dir
    )
    
    # Feature importance analysis
    sparse_weights = analyze_feature_importance(model_results, args.output_dir)
    
    # Analyze specific features
    if args.feature_indices:
        analyze_specific_features(
            sparse_before_pooling, df, args.feature_indices, args.output_dir
        )
    
    # Detailed analysis of top/bottom features
    print("\n" + "="*60)
    print("Detailed Analysis of Top/Bottom Features")
    print("="*60)
    
    # Get top 10 positive and bottom 10 negative features
    pos_features, neg_features = get_top_bottom_features(sparse_weights, n=10)
    
    print(f"\nTop 10 positive features: {pos_features}")
    print(f"Weights: {sparse_weights[pos_features]}")
    print(f"\nBottom 10 negative features: {neg_features}")
    print(f"Weights: {sparse_weights[neg_features]}")
    
    # Analyze each feature in detail
    REPORT_DATA['detailed_features']['positive'] = []
    REPORT_DATA['detailed_features']['negative'] = []
    
    for feat_idx in pos_features:
        feat_data = analyze_feature_detailed(
            feat_idx, sparse_before_pooling, df, args.output_dir,
            is_positive_weight=True,
            tm_extreme_fraction=args.tm_extreme_fraction,
            score_firing_threshold=args.score_firing_threshold,
        )
        REPORT_DATA['detailed_features']['positive'].append(feat_data)
    
    for feat_idx in neg_features:
        feat_data = analyze_feature_detailed(
            feat_idx, sparse_before_pooling, df, args.output_dir,
            is_positive_weight=False,
            tm_extreme_fraction=args.tm_extreme_fraction,
            score_firing_threshold=args.score_firing_threshold,
        )
        REPORT_DATA['detailed_features']['negative'].append(feat_data)
    
    # Generate HTML report
    generate_html_report(args.output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"HTML report: {os.path.join(args.output_dir, 'report.html')}")
    print("="*60)


if __name__ == '__main__':
    main()
