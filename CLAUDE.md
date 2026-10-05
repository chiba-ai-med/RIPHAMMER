# CLAUDE.md

## Project purpose

**RIPHAMMER** は、バルク Hi-C データから細胞型特異的な TAD 構造を抽出するための
Multi-band NMF（距離バンド分解 + H 共有結合 NMF）の Python パッケージである。

## Method: Multi-band NMF

Hi-C コンタクトマトリクスをゲノム距離 d ごとに 51 バンド（d=0..50, 100kb 解像度で 5Mb 以内）に分解し、
バンドごとに独立な基底行列 W_d、全バンドで共有される混合係数行列 H を持つ結合 NMF を実行する。

```
X_d ≈ W_d @ H   (d = 0, 1, ..., 50)
```

- per-band per-sample L1 正規化で距離減衰を等化（TAD 構造は保持）
- MU (Multiplicative Update) 則で最適化
- H 更新は全バンドの勾配を集約

## Package structure

```
riphammer/
  __init__.py       # バージョン、公開 API
  core.py           # MultibandNMF クラス（fit / transform / fold_in）
  io.py             # mcool → 特徴量ベクトル変換、W → cool 再構成
  tad.py            # cool → cooltools insulation → TAD 境界検出
  cli.py            # CLI エントリポイント（riphammer fit / riphammer tad）
pyproject.toml
tests/
```

## Naming convention

- 表示名・論文: RIPHAMMER（大文字）
- PyPI / pip install: `riphammer`（小文字）
- import: `import riphammer`（小文字）
- GitHub リポジトリ: `chiba-ai-med/RIPHAMMER`

## Design principles

- numpy ベースの軽量実装（GPU 不要）
- 100kb で数分、25kb で〜47 分、10kb で〜5 時間（15 サンプル）
- memmap + バンド逐次処理で 10kb 解像度まで対応可能
- 外部依存は numpy, cooler, cooltools, pandas 程度に抑える

## CLI interface (planned)

```bash
# Multi-band NMF の実行
riphammer fit --input data/ --k 8 --n-iter 500 --output results/

# NMF 基底からの TAD 検出
riphammer tad --input results/ --window 500000 --output tad/

# 既知 W への fold-in（混合比率推定）
riphammer fold-in --basis results/ --query sample.mcool --output foldin/
```

## Python API (planned)

```python
import riphammer

model = riphammer.MultibandNMF(n_components=8, max_iter=500)
model.fit(X, distances)
# model.W_bands: dict[int, ndarray]  — per-band basis
# model.H: ndarray                   — shared mixing coefficients

# TAD calling
boundaries = riphammer.call_tads(model.W_bands, features, window=500000)

# Fold-in
H_new = model.fold_in(X_new, distances_new)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Related repositories

- `chiba-ai-med/AIRE-experiments2` — RIPHAMMER を使った検証実験
- `chiba-ai-med/AIRE-experiments` — 前身（PoC 実験、Machima2 検証）
