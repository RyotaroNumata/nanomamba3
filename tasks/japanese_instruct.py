"""
Japanese instruction-following dataset.
https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset

~9M Japanese instruction-output pairs. We filter to keep only examples
where the output actually contains Japanese characters.
"""

from datasets import load_dataset
from tasks.common import Task


def _has_japanese(text):
    """Return True if text contains hiragana, katakana, or kanji."""
    for char in text:
        cp = ord(char)
        if (0x3040 <= cp <= 0x309F or   # Hiragana
                0x30A0 <= cp <= 0x30FF or   # Katakana
                0x4E00 <= cp <= 0x9FFF):    # CJK Unified Ideographs (kanji)
            return True
    return False


class JapaneseInstruct(Task):
    """
    Japanese instruction dataset filtered to Japanese-output examples only.
    After filtering, roughly 60% of the original ~9M rows remain.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "JapaneseInstruct only supports the train split"
        ds = load_dataset("izumi-lab/llm-japanese-dataset", split="train")
        self.ds = ds.filter(lambda row: _has_japanese(row.get("output", ""))).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"]
        inp = row.get("input", "")
        output = row["output"]

        user_content = f"{instruction}\n{inp}".strip() if inp else instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return {"messages": messages}
