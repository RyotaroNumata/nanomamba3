# nanochat

[English](README.md) | **日本語**

> **このフォークでは Mamba-3 MIMO を純粋な PyTorch でスクラッチ実装し、事前学習から英日バイリンガル SFT まで単一 GPU で完走しています。**
> Triton も mamba_ssm も使わず、SSM スキャン（`ssd_siso` / `ssd_mimo`）・データ依存減衰・MIMO rank-R 状態更新・部分 RoPE を [Mamba-3 論文](https://arxiv.org/abs/2603.15569) をベースに自前実装しました。nanochat の GPT Transformer と同じ学習ループにそのまま差し込める設計で、ONNX エクスポートによるオンデバイス推論にも対応しています。詳細は [nanochat/mamba3.py](nanochat/mamba3.py) と [runs/speedrun_mamba3_bilingual_sft.sh](runs/speedrun_mamba3_bilingual_sft.sh) を参照。

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は LLM をゼロからトレーニングするための、シンプルで最小限の実験ハーネスです。単一 GPU ノードで動作するよう設計されており、トークナイザー学習・事前学習・ファインチューニング（SFT / RL）・評価・推論・チャット UI まで LLM に必要なすべてのステージをカバーしています。たとえば、2019 年に約 4.3 万ドルかかった GPT-2 相当のモデルを、わずか約 48 ドル（8×H100 ノードで約 2 時間）でトレーニングし、おなじみの ChatGPT 風 Web UI で会話することができます。スポットインスタンスなら約 15 ドルまで下がります。

複雑さの唯一のダイヤルは `--depth`（Transformer 層数）です。その他のハイパーパラメータ（幅・ヘッド数・学習率・ステップ数・Weight Decay など）はすべて自動的に計算最適な値に設定されます。

**このフォークの追加機能**:
- `NANOCHAT_JA_RATIO` 環境変数を設定するだけで、英語と日本語を混合してトレーニング可能
- Mamba-3 MIMO rank-2 SSM アーキテクチャ対応（`--model-arch mamba3 --mamba-use-mimo`）

---

## Time-to-GPT-2 リーダーボード

現在の開発の主眼は、最も計算コストのかかる事前学習ステージの高速化です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトが常に GPT-2 相当モデルの学習方法のリファレンスです。

| # | 時間 | val_bpb | CORE | 説明 | 日付 | コミット | 貢献者 |
|---|------|---------|------|------|------|--------|--------|
| 0 | 168 時間 | - | 0.2565 | OpenAI GPT-2 オリジナル | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 ベースライン | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 + fp8 | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | バッチサイズ 1M トークンに拡大 | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | データセットを NVIDIA ClimbMix に変更 | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | 自動研究 round 1 | Mar 9 2026 | 6ed7d1d | @karpathy |
| 5 | 1.65 | 0.71800 | 0.2626 | 自動研究 round 2 | Mar 14 2026 | a825e63 | @karpathy |

主要指標は「Time-to-GPT-2」— 8×H100 ノードで GPT-2 (1.6B) の CORE スコア 0.256525 を超えるまでの経過時間です。

---

## クイックスタート

### GPT-2 を再現して会話する（Transformer）

パイプライン全体が [runs/speedrun.sh](runs/speedrun.sh) に収まっています。8×H100 ノードを起動してスクリプトを実行するだけです:

```bash
bash runs/speedrun.sh
```

約 3 時間かかるため `screen` セッションでの実行を推奨します。完了後:

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

ブラウザで表示された URL を開いてください（例: `http://209.20.xxx.xxx:8000/`）。

#### バイリンガル（英語 + 日本語）版

```bash
# 日本語比率 30% がデフォルト
bash runs/speedrun_bilingual.sh

# screen + wandb の場合
WANDB_RUN=bilingual screen -L -Logfile runs/speedrun_bilingual.log -S bilingual \
    bash runs/speedrun_bilingual.sh

# 日本語比率を変更する場合
NANOCHAT_JA_RATIO=0.5 bash runs/speedrun_bilingual.sh
```

---

### Mamba-3 MIMO（SSM アーキテクチャ）

Transformer の代わりに Mamba-3 MIMO rank-2 を使ったバイリンガル学習です。単一 GPU（RTX 3090 / 24GB VRAM 想定）で動作します。

```bash
bash runs/speedrun_mamba3_bilingual_sft.sh

# タグ指定 + wandb + screen
MODEL_TAG=mamba3_mimo_r2 WANDB_RUN=mamba3_mimo_r2 \
    screen -L -Logfile runs/mamba3_bilingual_sft.log -S mamba3sft \
    bash runs/speedrun_mamba3_bilingual_sft.sh
```

完了後（Mamba-3 は temperature=1.0 が推奨）:

```bash
source .venv/bin/activate
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_sft -t 1.0 -p "こんにちは！"
python -m scripts.chat_web
```

#### Transformer vs Mamba-3 MIMO スコア比較（SFT 後、d12 / 10k steps）

| タスク | Transformer (bilingual_v2) | Mamba-3 MIMO rank-2 |
|--------|---------------------------|---------------------|
| ARC-Easy | **36.45%** | 33.50% |
| ARC-Challenge | **33.28%** | 28.84% |
| MMLU | **31.89%** | 30.25% |
| GSM8K | **5.00%** | 1.06% |
| HumanEval | **9.15%** | 0.61% |
| SpellingBee | **99.22%** | 82.81% |
| JCommonsenseQA | **35.48%** | 33.24% |
| Base CORE | — | 0.1122 |
| ChatCORE | — | 0.1710 |

> Transformer と比べて全タスクで若干劣るが、SSM ならではの利点として **O(1) 推論メモリ**（KV キャッシュ不要）があり、長コンテキストで有利。

#### Transformer との主な違い

| 項目 | Transformer | Mamba-3 MIMO |
|------|-------------|--------------|
| `--model-arch` | `transformer`（デフォルト） | `mamba3 --mamba-use-mimo` |
| `--device-batch-size` | 8 | 2（VRAM 制限） |
| `--matrix-lr` | 0.003 | 0.001 |
| 推論メモリ | O(seq_len)（KV キャッシュ） | O(1)（固定サイズ状態） |
| 推奨 temperature | 0.6 | 1.0 |

#### 推論速度（RTX 3090、100 トークン生成、3回平均）

| バックエンド | prompt=32 | prompt=128 | prompt=512 | 備考 |
|------------|-----------|------------|------------|------|
| Mamba-3 MIMO — CUDA (PyTorch) | 51.1 tok/s | 49.1 tok/s | 51.1 tok/s | プロンプト長に**依存しない**（O(1) 再帰推論） |
| GPT Transformer — CUDA (PyTorch) | 97.7 tok/s | 99.6 tok/s | 81.6 tok/s | 短文は速いが長文で低下（KV キャッシュ肥大化） |
| Mamba-3 MIMO — CPU (PyTorch) | 21.9 tok/s | 20.8 tok/s | 17.7 tok/s | |
| Mamba-3 — ONNX fp32 (CPU) | 32.1 tok/s | 18.6 tok/s | 6.9 tok/s | 逐次プリフィル（トークンごと） |
| Mamba-3 — ONNX fp32 + chunk prefill (CPU) | 38.6 tok/s | 31.7 tok/s | 18.4 tok/s | chunk_size=32; 1.21×/1.71×/2.64× 高速化 |
| Mamba-3 — ONNX int8 (CPU) | 55.6 tok/s | 32.1 tok/s | 11.8 tok/s | fp32 比約 1.7× |
| Mamba-3 — ONNX int8 + chunk prefill (CPU) | **64.6 tok/s** | **48.3 tok/s** | **23.6 tok/s** | chunk_size=32; 1.17×/1.50×/2.00× 高速化；**CPU 全構成で最速** |

> SSM の O(1) メモリ特性により、Mamba-3 はコンテキストが長くなっても速度を維持します。
> chunk prefill は SSD スキャンを 32 トークン単位でバッチ処理するため、プロンプトが長いほど効果大（fp32: 2.64×、int8: 2.00× at 512 tokens）。
> 短文（prompt=32）では int8 + chunk prefill（64.6 tok/s）が CUDA PyTorch（51.1 tok/s）を上回ります。
> `python inference_bench.py --no-pytorch` で ONNX 数値を再現可能です。

#### ONNX エクスポートとオンデバイス推論

学習済み Mamba-3 SFT モデルを ONNX にエクスポートして CPU / モバイル向けに展開できます：

```bash
# fp32 デコードステップ + チャンクプリフィルモデルをエクスポート（推奨）
python -m scripts.export_onnx \
    --model-tag mamba3_mimo_r2_10k_sft \
    --fp32 \
    --output /tmp/mamba3_step_fp32.onnx \
    --export-prefill \
    --verify

# int8 に量子化（デコードとプリフィルを別々に）
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('/tmp/mamba3_step_fp32.onnx',         '/tmp/mamba3_step_int8.onnx',         weight_type=QuantType.QInt8)
quantize_dynamic('/tmp/mamba3_step_fp32_prefill.onnx', '/tmp/mamba3_step_int8_prefill.onnx', weight_type=QuantType.QInt8)
"

# ONNX でチャット（int8 + chunk prefill = CPU 最速構成）
python -m scripts.chat_onnx \
    --onnx /tmp/mamba3_step_int8.onnx \
    --onnx-prefill /tmp/mamba3_step_int8_prefill.onnx \
    -p "こんにちは！"

# インタラクティブモード
python -m scripts.chat_onnx \
    --onnx /tmp/mamba3_step_int8.onnx \
    --onnx-prefill /tmp/mamba3_step_int8_prefill.onnx
```

エクスポート後のファイルサイズ: fp32 デコード ~528 MB、fp32 プリフィル ~530 MB、int8 デコード ~133 MB、int8 プリフィル ~135 MB。

---

## セットアップ

### 前提条件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ
- GPU（CUDA 推奨。CPU / Apple Silicon でも動作しますが遅くなります）

### インストール

```bash
git clone https://github.com/<your-repo>/nanochat.git
cd nanochat
uv sync
source .venv/bin/activate
```

---

## 個別ステップの実行

### データセット

```bash
# 英語データ（ClimbMix-400B）
python -m nanochat.dataset -n 170

# 日本語データ（FineWeb-2-edu-japanese）
python -m nanochat.dataset -n 73 -l ja
```

### トークナイザー

```bash
python -m scripts.tok_train   # 学習
python -m scripts.tok_eval    # 評価
```

### 事前学習（Pretraining）

```bash
# 8× GPU ノード（推奨）
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26

# シングル GPU（自動でgrad accum）
python -m scripts.base_train -- --depth=26

# Mamba-3 MIMO（シングル GPU）
python -m scripts.base_train \
    --model-arch=mamba3 --mamba-use-mimo \
    --depth=12 --device-batch-size=2 --matrix-lr=0.001

# 評価
python -m scripts.base_eval
```

### SFT

```bash
python -m scripts.chat_sft    # 学習
python -m scripts.chat_eval   # 評価
```

### 会話

```bash
# CLI（-p を省略するとインタラクティブモード）
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI
python -m scripts.chat_web
```

### 研究・実験用クイックテスト（約 5 分）

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run="d12" --model-tag="d12" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```

---

## CPU / MPS での実行

[runs/runcpu.sh](runs/runcpu.sh) でシンプルな CPU / Apple Silicon デモが動作します。実用的な品質は出ませんが動作確認には便利です。

---

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに `COMPUTE_DTYPE`（`nanochat/common.py`）で精度を明示管理します。

| ハードウェア | デフォルト dtype | 理由 |
|-------------|----------------|------|
| CUDA SM80+（A100, H100） | `bfloat16` | ネイティブ bf16 テンソルコア |
| CUDA SM<80（V100, T4） | `float32` | bf16 なし（`NANOCHAT_DTYPE=float16` で GradScaler 有効化） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

環境変数で上書き可能:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train
```

---

## 環境変数一覧

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `NANOCHAT_JA_RATIO` | `0.0` | 日本語データの混合比率（0.0 = 英語のみ） |
| `NANOCHAT_BASE_DIR` | `~/.cache/nanochat` | データ・チェックポイントの保存先 |
| `NANOCHAT_DTYPE` | 自動検出 | 演算精度（`bfloat16` / `float32` / `float16`） |
| `WANDB_RUN` | `dummy` | W&B ログのラン名（`dummy` で無効化） |
| `OMP_NUM_THREADS` | — | マルチ GPU 時は `1` を推奨 |
| `PYTORCH_ALLOC_CONF` | — | `expandable_segments:True` で VRAM 断片化軽減 |

---

## ファイル構成

```
.
├── LICENSE
├── NOTICE                              # サードパーティライセンス表示
├── README.md                           # English README
├── README_ja.md                        # 日本語 README（このファイル）
├── dev
│   ├── gen_synthetic_data.py           # identity 会話データ生成例
│   ├── nanochat.png
│   └── repackage_data_reference.py
├── nanochat
│   ├── checkpoint_manager.py           # チェックポイント保存・読み込み
│   ├── common.py                       # 共通ユーティリティ / COMPUTE_DTYPE
│   ├── core_eval.py                    # DCLM CORE スコア評価
│   ├── dataloader.py                   # 分散対応データローダー
│   ├── dataset.py                      # データシャードのダウンロード
│   ├── engine.py                       # KV キャッシュ推論エンジン
│   ├── gpt.py                          # GPT Transformer（RoPE, GQA, SwiGLU）
│   ├── mamba3.py                       # Mamba-3 MIMO SSM（独自改変版）
│   ├── NOTICE                          # mamba3.py の原著作権表示と改変内容
│   ├── optim.py                        # AdamW + Muon オプティマイザ
│   └── tokenizer.py                    # BPE トークナイザー（32K 語彙）
├── runs
│   ├── speedrun.sh                     # GPT-2 スピードラン（8×H100）
│   ├── speedrun_bilingual.sh           # バイリンガル Transformer スピードラン
│   ├── speedrun_bilingual_sft.sh       # バイリンガル Transformer + SFT
│   ├── speedrun_mamba3.sh              # Mamba-3 事前学習のみ
│   ├── speedrun_mamba3_bilingual.sh    # Mamba-3 バイリンガル事前学習
│   └── speedrun_mamba3_bilingual_sft.sh # Mamba-3 バイリンガル + SFT（推奨）
├── scripts
│   ├── base_train.py                   # 事前学習
│   ├── base_eval.py                    # ベースモデル評価
│   ├── chat_sft.py                     # SFT 学習
│   ├── chat_eval.py                    # チャットモデル評価
│   ├── chat_cli.py                     # CLI チャット
│   ├── chat_web.py                     # Web UI チャット
│   ├── export_onnx.py                  # Mamba-3 → ONNX エクスポート
│   └── chat_onnx.py                    # ONNX モデルでのチャット
└── tasks
    ├── arc.py                          # ARC 評価タスク
    ├── gsm8k.py                        # GSM8K 数学問題
    ├── humaneval.py                    # Python コーディングタスク
    ├── jcommonsenseqa.py               # JCommonsenseQA（日本語）
    ├── japanese_instruct.py            # 日本語 instruction SFT タスク
    ├── mmlu.py                         # MMLU 多肢選択
    └── spellingbee.py                  # スペリング・文字カウントタスク
```

---

## テスト

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"          # 低速テストをスキップ
```

---

## コントリビューション

nanochat の目標は、1,000 ドル以下の予算でエンドツーエンドに扱えるマイクロモデルの水準を引き上げることです。アクセシビリティはコストだけでなく認知的複雑さにも関わります — nanochat は設定オブジェクトだらけのフレームワークではなく、単一の一貫したコードベースです。

AI 利用ポリシー: PR 提出時は、LLM が実質的に貢献した部分や自分が完全に理解していない部分を開示してください。

---

## 謝辞

- [nanoGPT](https://github.com/karpathy/nanoGPT)（事前学習のみをカバーした先行プロジェクト）
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)（リーダーボードのアイデアと実装の一部）
- [HuggingFace](https://huggingface.co/) — FineWeb, SmolTalk データセット
- [Lambda](https://lambda.ai/service/gpu-cloud) — 開発用コンピュート
- Mamba-3 実装の元となった [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal)（Apache 2.0, Copyright 2026 Vikram Karlex）

---

## 引用

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT

### サードパーティライセンス

[nanochat/mamba3.py](nanochat/mamba3.py) は [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal)（Copyright 2026 Vikram Karlex）を元に改変したもので、**Apache License 2.0** のもとで提供されます。詳細な帰属表示と改変内容は [nanochat/NOTICE](nanochat/NOTICE) を参照してください。
