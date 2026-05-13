"""
Japanese Dolly dataset (llm-jp/databricks-dolly-15k-ja).
https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja

Japanese translation of Databricks Dolly-15k by llm-jp, covering 7 task categories:
open_qa, closed_qa, brainstorming, classification,
information_extraction, creative_writing, summarization.

~15K high-quality single-turn instruction-output pairs.
Fields: instruction, context (optional), response, category.
"""

from datasets import load_dataset
from tasks.common import Task


class DollyJa(Task):
    """
    Japanese Dolly-15k (llm-jp version). ~15K single-turn instruction-output pairs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("llm-jp/databricks-dolly-15k-ja", split="train").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"]
        context = row.get("context", "") or ""
        response = row["response"]

        user_content = f"{instruction}\n\n{context}".strip() if context else instruction

        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}
