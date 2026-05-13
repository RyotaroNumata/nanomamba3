"""
Chat CLI using the exported ONNX Mamba3 decode-step model.

Usage:
    python -m scripts.chat_onnx --onnx /tmp/mamba3_step_fp32.onnx -p "日本の首都はどこですか？"
    python -m scripts.chat_onnx --onnx /tmp/mamba3_step_fp32.onnx  # interactive mode
"""

import argparse
import time
import numpy as np

from nanochat.tokenizer import get_tokenizer


def load_session(onnx_path):
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                providers=["CPUExecutionProvider"])
    return sess


def make_zero_states(sess, n_layer):
    """Infer state shapes from ONNX input metadata and create zero states."""
    meta = {inp.name: inp.shape for inp in sess.get_inputs()}
    states = {}
    for prefix in ("ssm", "k", "v", "angle"):
        for i in range(n_layer):
            name = f"{prefix}_{i}"
            shape = meta[name]
            states[name] = np.zeros(shape, dtype=np.float32)
    return states


def _update_states(sess_out, outputs) -> dict:
    """Rebuild states dict from ONNX outputs (strips 'new_' prefix)."""
    names = [o.name for o in sess_out][1:]
    return {name.removeprefix("new_"): outputs[i + 1] for i, name in enumerate(names)}


def step(sess, token_id: int, states: dict) -> tuple[np.ndarray, dict]:
    """Run one ONNX decode step. Returns (logits, new_states)."""
    feed = {"token_id": np.array([token_id], dtype=np.int64)}
    feed.update(states)
    outputs = sess.run(None, feed)
    return outputs[0], _update_states(sess.get_outputs(), outputs)


def prefill_chunk(sess_prefill, token_ids: list, states: dict) -> tuple[np.ndarray, dict]:
    """Run one chunk of tokens through the prefill ONNX model.
    token_ids must be exactly chunk_size tokens.
    Returns (logits_for_last_token, new_states).
    """
    feed = {"token_ids": np.array(token_ids, dtype=np.int64)}
    feed.update(states)
    outputs = sess_prefill.run(None, feed)
    return outputs[0], _update_states(sess_prefill.get_outputs(), outputs)


def infer_chunk_size(sess_prefill) -> int:
    """Read chunk_size from the prefill model's token_ids input shape."""
    for inp in sess_prefill.get_inputs():
        if inp.name == "token_ids":
            return inp.shape[0]
    raise ValueError("prefill model has no 'token_ids' input")


def sample(logits: np.ndarray, temperature: float = 0.3, top_k: int = 50) -> int:
    logits = logits.astype(np.float64)
    if top_k > 0:
        top_k = min(top_k, len(logits))
        threshold = np.sort(logits)[-top_k]
        logits[logits < threshold] = -np.inf
    if temperature > 0:
        logits /= temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))
    else:
        return int(np.argmax(logits))


def generate(sess, tokenizer, prompt_ids: list, max_new_tokens: int = 256,
             temperature: float = 0.3, top_k: int = 50, sess_prefill=None):
    """Prefill + decode using ONNX. Yields generated token ids."""
    # Infer n_layer from number of ssm_* inputs
    n_layer = sum(1 for inp in sess.get_inputs() if inp.name.startswith("ssm_"))
    states = make_zero_states(sess, n_layer)

    # Prefill: chunk-prefill if available, else token-by-token
    t0 = time.perf_counter()
    if sess_prefill is not None:
        chunk_size = infer_chunk_size(sess_prefill)
        n_chunked = (len(prompt_ids) // chunk_size) * chunk_size
        for i in range(0, n_chunked, chunk_size):
            logits, states = prefill_chunk(sess_prefill, prompt_ids[i:i + chunk_size], states)
        for tok in prompt_ids[n_chunked:]:
            logits, states = step(sess, tok, states)
    else:
        for tok in prompt_ids:
            logits, states = step(sess, tok, states)
    prefill_time = time.perf_counter() - t0

    # Decode
    next_id = sample(logits, temperature=0.0, top_k=1)  # greedy for first token
    for _ in range(max_new_tokens):
        yield next_id
        logits, states = step(sess, next_id, states)
        next_id = sample(logits, temperature=temperature, top_k=top_k)

    return prefill_time


def chat(sess, tokenizer, user_text: str, max_new_tokens: int = 256,
         temperature: float = 0.3, sess_prefill=None):
    """Format as chat conversation and generate response."""
    conversation = {"messages": [{"role": "user", "content": user_text}]}
    prompt_ids, _ = tokenizer.render_conversation(conversation)

    # Get special token ids to detect end of assistant turn
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    eos_tokens = {assistant_end} if assistant_end else set()

    print("Assistant: ", end="", flush=True)
    t0 = time.perf_counter()
    generated = []
    prev_text = ""
    n_layer = sum(1 for inp in sess.get_inputs() if inp.name.startswith("ssm_"))
    states = make_zero_states(sess, n_layer)

    # Prefill: chunk-prefill if available, else token-by-token
    if sess_prefill is not None:
        chunk_size = infer_chunk_size(sess_prefill)
        n_chunked = (len(prompt_ids) // chunk_size) * chunk_size
        for i in range(0, n_chunked, chunk_size):
            logits, states = prefill_chunk(sess_prefill, prompt_ids[i:i + chunk_size], states)
        for tok in prompt_ids[n_chunked:]:
            logits, states = step(sess, tok, states)
    else:
        for tok in prompt_ids:
            logits, states = step(sess, tok, states)

    # Decode
    next_id = sample(logits, temperature=0.0, top_k=1)
    for _ in range(max_new_tokens):
        if next_id in eos_tokens:
            break
        generated.append(next_id)
        # Stream: print only newly decoded characters (avoids \ufffd artifacts)
        text = tokenizer.decode(generated)
        if not text.endswith("\ufffd") and text != prev_text:
            print(text[len(prev_text):], end="", flush=True)
            prev_text = text
        logits, states = step(sess, next_id, states)
        next_id = sample(logits, temperature=temperature, top_k=50)

    elapsed = time.perf_counter() - t0
    print()
    tps = len(generated) / elapsed if elapsed > 0 else 0
    print(f"[{len(generated)} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]")
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="/tmp/mamba3_step_fp32.onnx")
    parser.add_argument("--onnx-prefill", type=str, default=None,
                        help="Path to chunk-prefill ONNX model (e.g. /tmp/mamba3_step_fp32_prefill.onnx)")
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("-t", "--temperature", type=float, default=0.3)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading ONNX model from {args.onnx}...")
    sess = load_session(args.onnx)

    sess_prefill = None
    if args.onnx_prefill:
        print(f"Loading chunk-prefill ONNX model from {args.onnx_prefill}...")
        sess_prefill = load_session(args.onnx_prefill)
        chunk_size = infer_chunk_size(sess_prefill)
        print(f"Chunk-prefill enabled (chunk_size={chunk_size})")

    tokenizer = get_tokenizer()

    n_layer = sum(1 for inp in sess.get_inputs() if inp.name.startswith("ssm_"))
    print(f"Model loaded: {n_layer} layers, temperature={args.temperature}")

    if args.prompt:
        chat(sess, tokenizer, args.prompt,
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
             sess_prefill=sess_prefill)
    else:
        print("Interactive mode (Ctrl+C to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                chat(sess, tokenizer, user_input,
                     max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature,
                     sess_prefill=sess_prefill)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
