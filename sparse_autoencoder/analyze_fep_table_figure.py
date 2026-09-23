#!/usr/bin/env python3
"""
FEP–SAE比較テーブル図 (Table 1) 生成

analyze_fep_table.py が出力する fep_sae_comparison_table.csv を、シンプルな
学術論文スタイルのテーブル画像（色なし・最小限の罫線）としてレンダリングする。
本文 Table 1 として採用された図。

生成物:
    figure/fep_table_figure.png -- Table 1 の画像（Nanobody, AHo position, Mutation,
                                    ddG_FEP, ddTm_pred の5列。Tm_pred WT/mut列は
                                    紙面の都合で図には含めず、元データCSVにのみ保持）

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_fep_table_figure.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Times New Romanは本環境に未インストールのため、字形・字幅互換のLiberation Serifを
# フォールバックに使用する（Times New Romanが利用可能な環境ではそちらが優先される）。
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = [
    'Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'Times', 'DejaVu Serif'
]
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # Times系に近い数式フォント

SCRIPT_DIR = Path(__file__).parent.resolve()
TABLE_PATH = SCRIPT_DIR / 'outputs/fep_table/fep_sae_comparison_table.csv'
OUTPUT_DIR = SCRIPT_DIR / 'outputs/fep_table/figure'

# 列定義: (ヘッダー行1, ヘッダー行2, データキー, フォーマット, 右寄せ?)
COLUMNS = [
    ('Nanobody',      '',                  'nanobody',      '{}',     False),
    ('AHo',           'position',          'aho_pos',       '{}',     True),
    ('Mutation',      '',                  'mutation',      '{}',     False),
    ('ΔΔG$_{FEP}$',  '(kcal mol$^{-1}$)', 'ddG_fep',      '{:+.2f}', True),
    ('ΔTm$_{pred}$', '(°C)',              'delta_tm_pred', '{:+.2f}', True),
]


def make_table_figure(df):
    df = df.copy()
    df['mutation'] = df.apply(lambda r: f"{r['wt_res']}→{r['mut_res']}", axis=1)

    n_rows = len(df)
    n_cols = len(COLUMNS)

    # 図サイズ・レイアウト定数
    COL_W   = [1.15, 0.65, 0.75, 1.30, 1.10]  # 各列幅 (inch)
    ROW_H   = 0.30   # データ行の高さ (inch)
    HEAD_H  = 0.52   # ヘッダー行の高さ (inch)
    PAD_T   = 0.10   # 上余白
    PAD_B   = 0.10   # 下余白

    fig_w = sum(COL_W)
    fig_h = PAD_T + HEAD_H + n_rows * ROW_H + PAD_B

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)

    # ── y座標（下端基準） ──────────────────────────────────────────────────
    y_top      = fig_h - PAD_T                    # テーブル上端
    y_head_bot = y_top - HEAD_H                   # ヘッダー下端 = データ上端
    y_row_tops = [y_head_bot - i * ROW_H for i in range(n_rows)]
    y_table_bot = PAD_B                           # テーブル下端

    # ── 列のx座標（左端累積） ─────────────────────────────────────────────
    col_x = []
    cx = 0.0
    for w in COL_W:
        col_x.append(cx)
        cx += w

    TEXT_PAD_L = 0.10   # 左寄せの左オフセット
    TEXT_PAD_R = 0.10   # 右寄せの右オフセット

    def xpos(ci, right):
        if right:
            return col_x[ci] + COL_W[ci] - TEXT_PAD_R
        return col_x[ci] + TEXT_PAD_L

    # ── 罫線（3本のみ） ───────────────────────────────────────────────────
    LINE_KW = dict(transform=ax.transData, color='black', clip_on=False)
    ax.plot([0, fig_w], [y_top,      y_top],      lw=1.2, **LINE_KW)   # テーブル上端
    ax.plot([0, fig_w], [y_head_bot, y_head_bot], lw=0.8, **LINE_KW)   # ヘッダー下
    ax.plot([0, fig_w], [y_table_bot, y_table_bot], lw=1.2, **LINE_KW) # テーブル下端

    # ── ヘッダー ──────────────────────────────────────────────────────────
    head_mid = (y_top + y_head_bot) / 2
    for ci, (h1, h2, _, _, right) in enumerate(COLUMNS):
        ha = 'right' if right else 'left'
        x  = xpos(ci, right)
        if h2:
            # 2行ヘッダー
            ax.text(x, head_mid + 0.09, h1, ha=ha, va='center',
                    fontsize=8.5, fontweight='bold', color='black')
            ax.text(x, head_mid - 0.09, h2, ha=ha, va='center',
                    fontsize=7.5, color='#444444')
        else:
            ax.text(x, head_mid, h1, ha=ha, va='center',
                    fontsize=8.5, fontweight='bold', color='black')

    # ── データ行 ─────────────────────────────────────────────────────────
    prev_nb = None
    for ri, (_, row) in enumerate(df.iterrows()):
        nb    = row['nanobody']
        y_mid = y_row_tops[ri] - ROW_H / 2

        # nanobody グループ間の細い区切り線
        if prev_nb is not None and nb != prev_nb:
            ax.plot([0, fig_w], [y_row_tops[ri], y_row_tops[ri]],
                    lw=0.4, color='#aaaaaa', ls='--', clip_on=False)

        for ci, (_, _, col_key, fmt, right) in enumerate(COLUMNS):
            ha = 'right' if right else 'left'
            x  = xpos(ci, right)

            if col_key == 'nanobody':
                # 同じ nanobody が続く行は名前を空白に
                text = nb if nb != prev_nb else ''
                ax.text(x, y_mid, text, ha=ha, va='center',
                        fontsize=8.5, fontstyle='italic', color='black')
            else:
                val  = row[col_key]
                text = fmt.format(val)
                ax.text(x, y_mid, text, ha=ha, va='center',
                        fontsize=8.5, color='black')

        prev_nb = nb

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE_PATH)

    fig  = make_table_figure(df)
    path = OUTPUT_DIR / 'fep_table_figure.png'
    fig.savefig(path, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"保存: {path}")


if __name__ == '__main__':
    main()
