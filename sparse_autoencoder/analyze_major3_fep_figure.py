#!/usr/bin/env python3
"""
追試3: FEP–SAE比較の論文用テーブル図生成

ΔΔG_FEP と SAE予測ΔTm をまとめた構造化テーブルをPNGで出力する。

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_major3_fep_figure.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).parent.resolve()
TABLE_PATH = SCRIPT_DIR / 'outputs/major3_fep_table/fep_sae_comparison_table.csv'
OUTPUT_DIR = SCRIPT_DIR / 'outputs/major3_fep_table/figure'

NANOBODY_ORDER = ['seq217', 'seq221', 'seq339', 'seq354', 'seq443']
NB_COLORS = {
    'seq217': '#1f77b4',
    'seq221': '#d95f02',
    'seq339': '#1b9e53',
    'seq354': '#c91a2a',
    'seq443': '#7552b0',
}

# テーブルの列定義: (表示名, データカラム名, フォーマット, 右寄せ?)
COLUMNS = [
    ('Nanobody',         'nanobody',      '{}',    False),
    ('AHo pos.',         'aho_pos',       '{}',    True),
    ('Mutation',         'mutation',      '{}',    False),
    ('ΔΔG_FEP\n(kcal mol⁻¹)', 'ddG_fep', '{:+.2f}', True),
    ('Tm_pred WT\n(°C)', 'tm_pred_wt',   '{:.1f}', True),
    ('Tm_pred mut\n(°C)','tm_pred_mut',  '{:.1f}', True),
    ('ΔTm_pred\n(°C)',   'delta_tm_pred','{:+.2f}', True),
]

# ΔΔG の不安定化/安定化カラー
def ddg_color(v):
    if v > 2.0:
        return '#fdd5d4'   # 強い不安定化: 薄赤
    elif v > 0:
        return '#fff5f0'   # 弱い不安定化: 極薄赤
    else:
        return '#edf8ee'   # 安定化: 薄緑

def dtm_color(v):
    if v > 0.3:
        return '#fdd5d4'   # SAEも正: 薄赤
    elif v < -0.2:
        return '#dceeff'   # SAEが負: 薄青
    else:
        return '#ffffff'   # 中立: 白


def build_table_data(df):
    """表示用データフレームを作成する。"""
    df = df.copy()
    df['mutation'] = df.apply(
        lambda r: f"{r['wt_res']}→{r['mut_res']}", axis=1)
    return df


def make_table_figure(df):
    df = build_table_data(df)

    n_rows = len(df)
    n_cols = len(COLUMNS)

    # 図サイズ: 列幅・行高さを手動調整
    COL_WIDTHS = [1.1, 0.65, 0.75, 1.25, 1.15, 1.15, 1.05]  # 各列の相対幅 (inch)
    ROW_H      = 0.42   # 1行の高さ (inch)
    HEAD_H     = 0.60   # ヘッダー行の高さ (inch)
    PAD        = 0.15   # 上下余白 (inch)

    fig_w = sum(COL_WIDTHS) + 0.3
    fig_h = HEAD_H + n_rows * ROW_H + PAD * 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # --- 列のx座標（左端基準）を計算 ---
    xs = [0.0]
    for w in COL_WIDTHS[:-1]:
        xs.append(xs[-1] + w)
    total_w = sum(COL_WIDTHS)

    # y座標: 上から順に配置
    # ヘッダー中央y
    head_y = fig_h - PAD - HEAD_H / 2
    # 各行の中央y
    row_ys = [fig_h - PAD - HEAD_H - (i + 0.5) * ROW_H for i in range(n_rows)]

    # --- ヘッダー行の背景 ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, fig_h - PAD - HEAD_H), total_w, HEAD_H,
        boxstyle='square,pad=0', transform=ax.transData,
        facecolor='#2c3e50', edgecolor='none', zorder=0,
        clip_on=False,
    ))

    # --- nanobody グループごとの背景帯（交互） ---
    group_starts = {}
    for nb in NANOBODY_ORDER:
        idxs = df.index[df['nanobody'] == nb].tolist()
        if idxs:
            group_starts[nb] = idxs

    gi = 0
    for nb in NANOBODY_ORDER:
        idxs = df.index[df['nanobody'] == nb].tolist()
        if not idxs:
            continue
        # 薄い背景（偶数グループ: わずかにグレー）
        bg = '#f7f7f7' if gi % 2 == 1 else '#ffffff'
        top_y = fig_h - PAD - HEAD_H - idxs[0] * ROW_H
        bot_y = fig_h - PAD - HEAD_H - (idxs[-1] + 1) * ROW_H
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, bot_y), total_w, top_y - bot_y,
            boxstyle='square,pad=0', transform=ax.transData,
            facecolor=bg, edgecolor='none', zorder=0,
            clip_on=False,
        ))
        gi += 1

    # --- ヘッダーテキスト ---
    for ci, (col_name, _, _, right) in enumerate(COLUMNS):
        ha = 'right' if right else 'left'
        xpos = xs[ci] + COL_WIDTHS[ci] - 0.05 if right else xs[ci] + 0.08
        ax.text(xpos, head_y, col_name,
                ha=ha, va='center',
                fontsize=8.5, fontweight='bold', color='white',
                transform=ax.transData, zorder=2, linespacing=1.4)

    # --- データ行 ---
    prev_nb = None
    for ri, (_, row) in enumerate(df.iterrows()):
        nb  = row['nanobody']
        y   = row_ys[ri]

        # nanobody セルは同じ nanobody の最初の行だけ表示
        for ci, (col_name, col_key, fmt, right) in enumerate(COLUMNS):
            if col_key == 'nanobody':
                if nb == prev_nb:
                    text = ''
                else:
                    text = nb
                    # nanobody名の左端に色ライン
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (xs[ci], y - ROW_H / 2), 0.06, ROW_H,
                        boxstyle='square,pad=0', transform=ax.transData,
                        facecolor=NB_COLORS[nb], edgecolor='none', zorder=1,
                        clip_on=False,
                    ))
            else:
                val  = row[col_key]
                text = fmt.format(val)

            # セル背景色（ΔΔG と ΔTm_pred）
            if col_key == 'ddG_fep':
                cell_bg = ddg_color(row['ddG_fep'])
                ax.add_patch(mpatches.FancyBboxPatch(
                    (xs[ci], y - ROW_H / 2 + 0.02),
                    COL_WIDTHS[ci] - 0.04, ROW_H - 0.04,
                    boxstyle='round,pad=0.02', transform=ax.transData,
                    facecolor=cell_bg, edgecolor='none', zorder=1,
                    clip_on=False,
                ))
            elif col_key == 'delta_tm_pred':
                cell_bg = dtm_color(row['delta_tm_pred'])
                ax.add_patch(mpatches.FancyBboxPatch(
                    (xs[ci], y - ROW_H / 2 + 0.02),
                    COL_WIDTHS[ci] - 0.04, ROW_H - 0.04,
                    boxstyle='round,pad=0.02', transform=ax.transData,
                    facecolor=cell_bg, edgecolor='none', zorder=1,
                    clip_on=False,
                ))

            ha   = 'right' if right else 'left'
            xpos = xs[ci] + COL_WIDTHS[ci] - 0.10 if right else xs[ci] + 0.14
            color = NB_COLORS[nb] if col_key == 'nanobody' and text else '#1a1a1a'
            ax.text(xpos, y, text,
                    ha=ha, va='center',
                    fontsize=8.5, color=color,
                    fontweight='bold' if col_key == 'nanobody' else 'normal',
                    transform=ax.transData, zorder=2)

        prev_nb = nb

    # --- 水平区切り線 ---
    # ヘッダー下
    lw_head = 1.5
    ax.plot([0, total_w],
            [fig_h - PAD - HEAD_H, fig_h - PAD - HEAD_H],
            color='white', lw=lw_head, transform=ax.transData,
            zorder=3, clip_on=False)
    # 各行の下の細線
    for ri in range(n_rows):
        y_line = row_ys[ri] - ROW_H / 2
        ax.plot([0, total_w], [y_line, y_line],
                color='#cccccc', lw=0.5, transform=ax.transData,
                zorder=3, clip_on=False)
    # nanobody 境界線（太め）
    prev_nb = None
    for ri, (_, row) in enumerate(df.iterrows()):
        if prev_nb is not None and row['nanobody'] != prev_nb:
            y_line = row_ys[ri] + ROW_H / 2
            ax.plot([0, total_w], [y_line, y_line],
                    color='#888888', lw=1.0, transform=ax.transData,
                    zorder=4, clip_on=False)
        prev_nb = row['nanobody']

    # --- 外枠 ---
    bot_y = row_ys[-1] - ROW_H / 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, bot_y), total_w, fig_h - PAD - bot_y,
        boxstyle='square,pad=0', transform=ax.transData,
        facecolor='none', edgecolor='#555555', linewidth=1.2,
        zorder=5, clip_on=False,
    ))

    # --- 凡例（ΔΔG色の説明）---
    legend_items = [
        (mpatches.Patch(facecolor='#fdd5d4', edgecolor='#aaaaaa', linewidth=0.5),
         'ΔΔG > +2.0  (strongly destabilizing)'),
        (mpatches.Patch(facecolor='#fff5f0', edgecolor='#aaaaaa', linewidth=0.5),
         '0 < ΔΔG ≤ +2.0  (mildly destabilizing)'),
        (mpatches.Patch(facecolor='#edf8ee', edgecolor='#aaaaaa', linewidth=0.5),
         'ΔΔG < 0  (stabilizing)'),
        (mpatches.Patch(facecolor='#dceeff', edgecolor='#aaaaaa', linewidth=0.5),
         'ΔTm_pred < −0.2  (SAE: destabilizing)'),
    ]
    legend = ax.legend(
        handles=[h for h, _ in legend_items],
        labels=[l for _, l in legend_items],
        loc='upper right',
        bbox_to_anchor=(total_w, bot_y - 0.05),
        bbox_transform=ax.transData,
        fontsize=7.5, framealpha=0.9,
        title='Cell color key', title_fontsize=7.5,
        frameon=True,
    )

    ax.set_xlim(-0.1, total_w + 0.1)
    ax.set_ylim(bot_y - 0.5, fig_h + 0.05)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE_PATH)

    fig = make_table_figure(df)
    path = OUTPUT_DIR / 'fep_table_figure.png'
    fig.savefig(path, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"テーブル図保存: {path}")


if __name__ == '__main__':
    main()
