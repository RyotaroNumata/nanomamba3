"""
Japanese SFT model evaluation: run fixed Japanese prompts through the chat model
and display Q/A pairs clearly.

Examples:
    # Run default Japanese prompts
    python -m scripts.ja_chat_eval

    # Custom prompts from command line
    python -m scripts.ja_chat_eval -p "日本の首都はどこですか？" -p "自己紹介してください。"

    # Load prompts from a text file (one prompt per line)
    python -m scripts.ja_chat_eval --prompt-file prompts.txt

    # Adjust generation parameters
    python -m scripts.ja_chat_eval --temperature 0.8 --max-tokens 200
"""

import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.report import get_report

# ---------------------------------------------------------------------------
# Default Japanese evaluation prompts

DEFAULT_PROMPTS = [
    # 知識・事実
    "日本の首都はどこですか？",
    "水の化学式を教えてください。",
    "富士山の高さは何メートルですか？",
    # 日常会話
    "自己紹介をしてください。",
    "今日のおすすめの過ごし方を教えてください。",
    # 推論・説明
    "なぜ空は青いのですか？",
    "人工知能とは何ですか？簡単に説明してください。",
    # 指示に従うタスク
    "「ありがとう」を英語、中国語、韓国語で言ってください。",
    "1から10までの偶数を列挙してください。",
]

# ---------------------------------------------------------------------------

def run_prompt(engine, tokenizer, special_tokens, prompt, max_tokens, temperature, top_k):
    """Run a single prompt through the SFT model and return the response string."""
    bos, user_start, user_end, assistant_start, assistant_end = special_tokens

    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(prompt))
    tokens.append(user_end)
    tokens.append(assistant_start)

    response_tokens = []
    for token_column, _ in engine.generate(tokens, num_samples=1,
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_k=top_k):
        token = token_column[0]
        response_tokens.append(token)

    if response_tokens and response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)

    # Decode, strip the trailing <|assistant_end|> for clean display
    response = tokenizer.decode(response_tokens)
    response = response.replace(tokenizer.decode([assistant_end]), "").strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="Japanese SFT evaluation")
    parser.add_argument("-p", "--prompt", action="append", dest="prompts", default=[],
                        help="Japanese prompt (can be specified multiple times)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Text file with one prompt per line")
    parser.add_argument("-i", "--source", type=str, default="sft",
                        help="Model source: sft|rl (default: sft)")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load (default: last)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    # Determine prompts to use
    prompts = list(args.prompts)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if not prompts:
        prompts = DEFAULT_PROMPTS

    # Model setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    special_tokens = (
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|user_start|>"),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
        tokenizer.encode_special("<|assistant_end|>"),
    )

    step_info = meta.get("step", "?")
    print(f"\nModel : {args.source} (step {step_info})")
    print(f"Params: temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
    print("=" * 60)

    results = []
    for prompt in prompts:
        response = run_prompt(engine, tokenizer, special_tokens, prompt,
                              args.max_tokens, args.temperature, args.top_k)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
        print("-" * 60)
        results.append({"Q": prompt, "A": response})

    # Save to report
    report_data = [{"model": args.source, "step": step_info}]
    report_data.extend(results)
    get_report().log(section="Japanese chat evaluation", data=report_data)
    print("\nResults logged to report.")


if __name__ == "__main__":
    main()
