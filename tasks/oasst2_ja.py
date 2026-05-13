"""
Japanese OASST2 dataset (llm-jp/oasst2-33k-ja).
https://huggingface.co/datasets/llm-jp/oasst2-33k-ja

Human-annotated multi-turn Japanese conversations derived from OpenAssistant2.
~33K conversations. Higher quality than machine-generated datasets.
Field: conversations (list of {"role": "user"/"assistant", "content": "..."})
"""

from datasets import load_dataset
from tasks.common import Task


class OASST2Ja(Task):
    """
    Japanese OASST2 conversations. ~33K multi-turn examples.
    Each example is a full conversation tree flattened to a single path.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("llm-jp/oasst2-33k-ja", split="train").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        conversations = row["conversations"]  # list of {"role", "content"}

        # Filter to user/assistant only (drop system messages if present)
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversations
            if msg["role"] in ("user", "assistant")
        ]

        # Must start with user and alternate correctly
        if not messages or messages[0]["role"] != "user":
            # Fallback: skip malformed examples by returning an empty stub
            # (TaskMixture will still call this, so return minimal valid conversation)
            messages = [
                {"role": "user",      "content": "こんにちは"},
                {"role": "assistant", "content": "こんにちは！何かお手伝いできることはありますか？"},
            ]

        return {"messages": messages}
