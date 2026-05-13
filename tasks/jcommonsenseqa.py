"""
JCommonsenseQA from the JGLUE benchmark.
https://huggingface.co/datasets/sbintuitions/JCommonsenseQA

Japanese commonsense question answering with 5 choices.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


class JCommonsenseQA(Task):
    """
    5-choice Japanese commonsense QA. eval_type = 'categorical'.
    train: 8,939 examples. validation: 1,119 examples.
    """

    def __init__(self, split="validation", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "JCommonsenseQA split must be train|validation"
        self.ds = load_dataset("sbintuitions/JCommonsenseQA", split=split).shuffle(seed=42)
        self.letters = ["A", "B", "C", "D", "E"]

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = [row[f"choice{i}"] for i in range(5)]
        answer_letter = self.letters[row["label"]]  # label is 0-4

        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_letter},
        ]
        return {
            "messages": messages,
            "letters": self.letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], \
            f"JCommonsenseQA answer '{assistant_response}' must be one of {conversation['letters']}"
        expected = conversation["messages"][-1]["content"]
        return assistant_response == expected
