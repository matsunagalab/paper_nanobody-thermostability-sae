#!/usr/bin/env python3
"""
追試3: FEP–SAE予測Tm比較テーブル (Reviewer 2 Major 3)

既存FEP計算のΔΔGと、SAE Ridge回帰モデルによる予測ΔTmを対応づけたテーブルを生成する。

FEP nanobody (seq217,221,339,354,443) はvhh_thermo_seq.csvに含まれないため：
  1. ddG.csvの変異配列から1残基を元に戻してWT配列を再構築
  2. WT+変異配列に対してESM-2 SFT + SAEで埋め込みを生成
  3. nbbench訓練データでfit したMaxAbsScaler+RidgeCVで予測Tmを計算

実行方法（sparse_autoencoder/ ディレクトリから）:
    python analyze_major3_fep_table.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import RidgeCV

# ── パス設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR / '..'

FEP_ROOT    = REPO_ROOT / 'fep'
SPARSE_DIR  = SCRIPT_DIR / 'interplm/nbbench/nbbench_embedding_sparse_sft_8m_expansion_32_lr_9e-5_l1_7e-2'
TM_DATA     = REPO_ROOT / 'data/nbbench/thermo-seq/vhh_thermo_seq.csv'
TRAIN_CSV   = REPO_ROOT / 'data/nbbench/thermo-seq/train.csv'
ESM_WEIGHTS = REPO_ROOT / 'supervised_finetuning/models/sft_esm2_8m_optuna/encoder/converted_model.pt'
SAE_WEIGHTS = SCRIPT_DIR / 'models/sft_8m_100k/expansion_32_lr_9e-5_l1_7e-2/layer_6/ae.pt'
OUTPUT_DIR  = SCRIPT_DIR / 'outputs/major3_fep_table'

LAYER    = 6
CV_FOLDS = 10

FEP_SEQS = [217, 221, 339, 354, 443]

# ddG.csv の列定義
# col0: ungapped_pos, col1: WT_res, col2: mut_res, col3: ddG_kcal,
# col4: aho_pos, col5: another_num, col6: mutant_seq
COL_NAMES = ['ungapped_pos', 'wt_res', 'mut_res', 'ddG_fep', 'aho_pos', 'num2', 'mutant_seq']


# ── FEPデータ読み込みとWT配列再構築 ──────────────────────────────────────────

def reconstruct_wt(row: pd.Series) -> str:
    """変異配列の1残基を元に戻してWT配列を再構築する。
    ungapped_pos は1-indexed。"""
    mutant_seq = row['mutant_seq']
    pos = int(row['ungapped_pos']) - 1   # 0-indexed
    wt_res  = row['wt_res']
    mut_res = row['mut_res']
    assert mutant_seq[pos] == mut_res, (
        f"位置{pos+1}: expected {mut_res}, got {mutant_seq[pos]}")
    return mutant_seq[:pos] + wt_res + mutant_seq[pos+1:]


def load_fep_data() -> pd.DataFrame:
    """全FEP nanobodyのddG.csvを読み込み、WT配列を再構築して返す。"""
    records = []
    for seq_id in FEP_SEQS:
        csv_path = FEP_ROOT / f'seq{seq_id}/ionized-FEP/ddG.csv'
        df = pd.read_csv(csv_path, header=None, names=COL_NAMES)
        df['nanobody'] = f'seq{seq_id}'

        # WT配列を1行目から再構築（全行で同一のはず）
        wt_seq = reconstruct_wt(df.iloc[0])
        # 2行目以降でも一致を確認
        for _, r in df.iterrows():
            wt_check = reconstruct_wt(r)
            assert wt_check == wt_seq, \
                f"seq{seq_id}: WT配列が行によって一致しません\n  row: {wt_check}\n  ref: {wt_seq}"
        df['wt_seq'] = wt_seq
        records.append(df)

    fep_df = pd.concat(records, ignore_index=True)
    print(f"FEPデータ: {len(fep_df)}変異 × {len(FEP_SEQS)} nanobody")
    for seq_id in FEP_SEQS:
        sub = fep_df[fep_df['nanobody'] == f'seq{seq_id}']
        print(f"  seq{seq_id}: WT={sub.iloc[0]['wt_seq'][:40]}... (len={len(sub.iloc[0]['wt_seq'])})")
    return fep_df


# ── ESM-2 + SAE 埋め込み生成 ─────────────────────────────────────────────────

def load_esm_model(device: torch.device):
    """SFT ESM-2 8Mモデルを読み込む。"""
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    ckpt = torch.load(ESM_WEIGHTS, map_location=device)
    # ESM2_TmRegressor形式なら 'esm.' プレフィクスを除去
    if any(k.startswith('esm.') for k in ckpt.keys()):
        ckpt = {k[4:]: v for k, v in ckpt.items() if k.startswith('esm.')}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"ESM-2 SFT重みを読み込み (missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device).eval()
    return model, alphabet


def embed_sequence(model, alphabet, seq: str, device: torch.device) -> torch.Tensor:
    """1配列のESM-2 layer-6活性化（残基次元、特殊トークン除去済み）を返す。
    shape: (seq_len, 320)"""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([('seq', seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        result = model(tokens, repr_layers=[LAYER])
    act = result['representations'][LAYER]   # (1, seq_len+2, 320)
    seq_len_with_special = (tokens[0] != alphabet.padding_idx).sum()
    return act[0, 1:seq_len_with_special - 1].cpu()   # 特殊トークン除去


def load_sae(device: torch.device):
    """SFT SAEモデルを読み込む。"""
    sys.path.insert(0, str(SCRIPT_DIR))
    from interplm.sae.inference import load_sae as _load_sae
    sae = _load_sae(SAE_WEIGHTS, device=str(device))
    return sae


def sae_encode(sae, esm_act: torch.Tensor, device: torch.device) -> torch.Tensor:
    """ESM活性化 (seq_len, 320) → SAE sparse活性化 (seq_len, 10240)"""
    with torch.no_grad():
        x = esm_act.to(device)
        feat = torch.nn.functional.relu(
            (x - sae.bias) @ sae.encoder.weight.T + sae.encoder.bias
        )
    return feat.cpu()


def generate_fep_embeddings(fep_df: pd.DataFrame) -> dict[str, torch.Tensor]:
    """WT・変異配列全てのSAEスパース活性化を生成する。
    返り値: {seq_label: (seq_len, 10240)} の辞書"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n埋め込み生成デバイス: {device}")

    model, alphabet = load_esm_model(device)
    sae = load_sae(device)

    embeddings = {}

    # WT配列（nanobody毎に1つ）
    wt_done = set()
    for _, row in fep_df.iterrows():
        nb = row['nanobody']
        if nb in wt_done:
            continue
        print(f"  WT {nb}: {row['wt_seq'][:30]}...")
        esm_act = embed_sequence(model, alphabet, row['wt_seq'], device)
        sparse  = sae_encode(sae, esm_act, device)
        embeddings[f'{nb}_WT'] = sparse
        wt_done.add(nb)

    # 変異配列
    for _, row in fep_df.iterrows():
        nb  = row['nanobody']
        pos = int(row['ungapped_pos'])
        mut = row['mut_res']
        label = f"{nb}_p{pos}{row['wt_res']}{mut}"
        print(f"  mut {label}: {row['mutant_seq'][:30]}...")
        esm_act = embed_sequence(model, alphabet, row['mutant_seq'], device)
        sparse  = sae_encode(sae, esm_act, device)
        embeddings[label] = sparse

    return embeddings


# ── Ridge モデルの学習 ────────────────────────────────────────────────────────

def fit_ridge():
    """nbbench訓練データでMaxAbsScaler+RidgeCVをfit。"""
    df = pd.read_csv(TM_DATA)
    sparse_proteins = torch.load(
        SPARSE_DIR / f'all_sequences_layer_{LAYER}_sparse_activations.pt', map_location='cpu'
    )
    y = df['label'].values
    train_df = pd.read_csv(TRAIN_CSV)
    seq2i = {s: i for i, s in enumerate(df['seq'])}
    train_idx = np.array([seq2i[s] for s in train_df['seq']], dtype=int)

    X = torch.stack([p.mean(dim=0) for p in sparse_proteins]).numpy()
    y_tr = y[train_idx]
    X_tr = X[train_idx]

    scaler = MaxAbsScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    ridge = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=CV_FOLDS,
                    scoring='neg_root_mean_squared_error')
    ridge.fit(X_tr_sc, y_tr)
    print(f"Ridge α={ridge.alpha_:.4f}")
    return scaler, ridge


def predict_tm(scaler, ridge, sparse_act: torch.Tensor) -> float:
    """sparse活性化 (seq_len, 10240) → 予測Tm"""
    x = sparse_act.mean(dim=0).numpy().reshape(1, -1)
    x_sc = scaler.transform(x)
    return float(ridge.predict(x_sc)[0])


# ── テーブル構築 ──────────────────────────────────────────────────────────────

def build_table(fep_df: pd.DataFrame, embeddings: dict, scaler, ridge) -> pd.DataFrame:
    rows = []
    for _, row in fep_df.iterrows():
        nb  = row['nanobody']
        pos = int(row['ungapped_pos'])
        mut = row['mut_res']
        wt  = row['wt_res']

        wt_key  = f'{nb}_WT'
        mut_key = f'{nb}_p{pos}{wt}{mut}'

        tm_wt  = predict_tm(scaler, ridge, embeddings[wt_key])
        tm_mut = predict_tm(scaler, ridge, embeddings[mut_key])

        rows.append({
            'nanobody':      nb,
            'ungapped_pos':  pos,
            'aho_pos':       int(row['aho_pos']),
            'wt_res':        wt,
            'mut_res':       mut,
            'ddG_fep':       round(float(row['ddG_fep']), 3),
            'tm_pred_wt':    round(tm_wt,  2),
            'tm_pred_mut':   round(tm_mut, 2),
            'delta_tm_pred': round(tm_mut - tm_wt, 2),
        })

    return pd.DataFrame(rows)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. FEPデータ読み込み・WT再構築
    print("=== FEPデータ読み込み ===")
    fep_df = load_fep_data()

    # 2. 埋め込み生成
    print("\n=== 埋め込み生成 ===")
    embeddings = generate_fep_embeddings(fep_df)

    # 3. Ridge学習
    print("\n=== Ridge学習 (nbbench train) ===")
    scaler, ridge = fit_ridge()

    # 4. テーブル構築
    print("\n=== テーブル構築 ===")
    result_df = build_table(fep_df, embeddings, scaler, ridge)

    # 保存
    out_path = OUTPUT_DIR / 'fep_sae_comparison_table.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\nテーブル保存: {out_path}")

    # 表示
    print("\n=== 結果テーブル ===")
    print(result_df.to_string(index=False))

    # 論文用整形テーブル（nanobody毎にWT Tmを先頭行に）
    print("\n=== nanobody毎のサマリー ===")
    for nb in [f'seq{i}' for i in FEP_SEQS]:
        sub = result_df[result_df['nanobody'] == nb]
        if sub.empty:
            continue
        tm_wt = sub.iloc[0]['tm_pred_wt']
        print(f"\n{nb} (Tm_pred_WT = {tm_wt:.1f} °C):")
        print(f"  {'AHo':>4}  {'WT':>2}→{'Mut':>3}  {'ΔΔG_FEP':>9}  {'ΔTm_pred':>9}")
        for _, r in sub.iterrows():
            print(f"  {r['aho_pos']:>4}  {r['wt_res']:>2}→{r['mut_res']:>3}  "
                  f"{r['ddG_fep']:>9.3f}  {r['delta_tm_pred']:>9.2f}")

    print(f"\n完了。出力先: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
