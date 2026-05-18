# nanomamba3 — ピュアMamba3言語モデルをゼロから作る

[English](README_en.md) | **日本語**

このレポジトリは、**Mamba3 を使って、英日バイリンガル言語モデルをゼロから作るプロジェクトです。**

nanochatをベースとして、トークナイザーの学習から、事前学習、SFT（Supervised Fine-Tuning）、評価、推論まで、LLM に必要な全ステージを一つのリポジトリでカバーしています。Mamba3のモデル実装は、SSM スキャンから MIMO rank-R 状態更新、部分 RoPE まで **PyTorchだけ** でスクラッチ実装しました。

また、本レポジトリで作成する言語モデルは**ピュアMamba3アーキテクチャ**の言語モデルになっています。  
VRAM24GBのGPUが1枚あれば作れるので、みなさん是非自分だけのMamba3LMを作りませんか？

## アーキテクチャ：Mamba 3とは

Mamba-3 は State Space Model（SSM）をベースにした系列モデルです。固定サイズの隠れ状態 `h_t` を再帰的に更新することで、**推論メモリ O(1)** を実現しています。

**MIMO（Multi-Input Multi-Output）rank-R** はチャネル間の相関を学習できるよう SSM を行列に拡張したもので、本実装では rank=2 を使用しています。 \
実装は [nanochat/mamba3.py](nanochat/mamba3.py) にあります。


## モデルの重み

学習済みモデルの重みは HuggingFace で公開しています：

**[kikyo0114/nanochat-mamba3-mimo-r2](https://huggingface.co/kikyo0114/nanochat-mamba3-mimo-r2)**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "kikyo0114/nanochat-mamba3-mimo-r2",
    local_dir="~/.cache/nanochat/chatsft_checkpoints/mamba3_mimo_r2_10k_sft",
)
```

---

## クイックスタート
### セットアップ

```bash
git clone https://github.com/RyotaroNumata/nanomamba3.git
cd nanomamba3
uv sync
source .venv/bin/activate
```

### Mamba 3 事前学習（シングル GPU）
下記実行でデータのダウンロードからトークナイザ作成、事前学習、SFTまで一気に実施できます！

```bash
bash runs/speedrun_mamba3_bilingual_sft.sh
```

内部では以下の順に実行されます：
1. 事前学習（~10,000 steps）
2. SFT（SmolTalk + 日本語インストラクションデータ）

### 各実験を個別にやりたい場合
#### データのダウンロード

```bash
# 英語データ（ClimbMix-400B）
python -m nanochat.dataset -n 170

# 日本語データ（FineWeb-2-edu-japanese）
python -m nanochat.dataset -n 73 -l ja
```

#### トークナイザー学習

```bash
python -m scripts.tok_train
python -m scripts.tok_eval   # 確認
```

#### 事前学習
NANOCHAT_JA_RATIOで日本語データの混合率を制御できます。
```bash
WANDB_RUN=mamba3_pretrain NANOCHAT_JA_RATIO=0.3 python -m scripts.base_train \
    --model-arch=mamba3 \
    --mamba-use-mimo \
    --depth=12 \
    --device-batch-size=2 \
    --matrix-lr=0.001 \
    --run=mamba3_pretrain \
    --model-tag=mamba3_mimo_r2

# 評価
python -m scripts.base_eval --device-batch-size=2 --model-tag=mamba3_mimo_r2
```

#### SFT

```bash
WANDB_RUN=mamba3_sft NANOCHAT_JA_RATIO=0.3 python -m scripts.chat_sft \
    --device-batch-size=8 \
    --model-tag=mamba3_mimo_r2 \
    --output-tag=mamba3_mimo_r2_sft \
    --run=mamba3_sft

# 評価
python -m scripts.chat_eval -i sft --model-tag=mamba3_mimo_r2_sft
```

### チャット
下記実行でチャット形式で会話できます！
```bash
# CLI（temperature=1.0 推奨）
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_10k_sft -t 1.0 -p "こんにちは！"

# Web UI（ブラウザで localhost:8000）
python -m scripts.chat_web -g mamba3_mimo_r2_10k_sft -t 1.0
```

> **temperature について**：Mamba-3 は固定サイズ状態への圧縮という構造上の制約から、低温では繰り返しループが起きやすい傾向があります。temperature=1.0 での使用を推奨します。

---

## ファイル構成

```
nanochat/
├── mamba3.py              # Mamba-3 MIMO SSM 実装（本プロジェクトの中核）
├── gpt.py                 # GPT Transformer（比較用・同じ学習ループで動く）
├── engine.py              # 推論エンジン（KV キャッシュ / SSM 再帰推論）
├── tokenizer.py           # BPE トークナイザー（32K 語彙、英日対応）
├── dataloader.py          # 分散対応データローダー
├── dataset.py             # データシャードのダウンロード
├── optim.py               # AdamW + Muon オプティマイザ
├── checkpoint_manager.py  # チェックポイント保存・読み込み
├── common.py              # COMPUTE_DTYPE、DDP 設定
└── NOTICE                 # mamba3.py の原著作権・改変内容

scripts/
├── base_train.py          # 事前学習メインループ
├── base_eval.py           # ベースモデル評価（CORE + BPB）
├── chat_sft.py            # SFT 学習
├── chat_eval.py           # チャットモデル評価
├── chat_cli.py            # CLI チャット
└── chat_web.py            # Web UI チャット

runs/
├── speedrun_mamba3_bilingual_sft.sh  # Mamba-3 フルパイプライン（推奨）
├── speedrun_mamba3_bilingual.sh      # Mamba-3 事前学習のみ
└── speedrun.sh                       # GPT-2 スピードラン（Transformer 版）
```

---

## 学習データ

### 事前学習データ

- **英語**：[ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- **日本語**：[FineWeb-2 Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese)（[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) を元に作成、[ODC-By v1.0](https://opendatacommons.org/licenses/by/1.0/) ライセンス）
  - 元ウェブデータ：[CommonCrawl](https://commoncrawl.org)（[利用規約](https://commoncrawl.org/terms-of-use)）

**帰属表示（ODC-By v1.0 要件）**：FineWeb-2 Japanese を使用した成果物を公開する場合は、FineWeb2 および CommonCrawl の帰属表示が必要です。

### SFT データ

- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)（英語）
- [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)（日本語会話）
- [Dolly-JA](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
- [Magpie-JA](https://huggingface.co/datasets/Aratako/Magpie-Qwen2.5-72B-Instruct-Japanese-300K-Filtered)


## ライセンス

MIT

[nanochat/mamba3.py](nanochat/mamba3.py) は [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal)（Copyright 2026 Vikram Karlex）を元に改変したもので、**Apache License 2.0** のもとで提供されます。詳細は [nanochat/NOTICE](nanochat/NOTICE) を参照してください。

---

## 謝辞

- [nanochat](https://github.com/karpathy/nanochat)（ベースとなったLLMトレーニングフレームワーク by Andrej Karpathy）
- [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal)（Mamba-3 実装の元）
