# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Training on the model's answers only.

    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
                                      response_part="<|im_start|>assistant\\n")

The loss then counts only the tokens after each response_part, up to the next
instruction_part: the system prompt and the user's turns are context, not
targets. The masking is done on each collated batch, so it applies the same
way to padded and padding-free batches and with any TRL version; in a
padding-free row each example, from one position id 0 to the next, is masked
on its own.
"""

import torch


def _find(row: list, part: list, start: int = 0) -> int:
    """The index after the first match of part in row[start:], or -1."""
    n = len(part)
    for i in range(start, len(row) - n + 1):
        if row[i:i + n] == part:
            return i + n
    return -1


def response_mask(ids: list, instruction: list, response: list) -> list:
    """True where a token belongs to an answer: after a response part, before the next instruction part."""
    mask = [False] * len(ids)
    at = 0
    while True:
        begin = _find(ids, response, at)
        if begin < 0:
            return mask
        end = _find(ids, instruction, begin)
        stop = end - len(instruction) if end >= 0 else len(ids)
        for i in range(begin, stop):
            mask[i] = True
        if end < 0:
            return mask
        at = stop


class ResponsesOnly:
    """A data collator that sets the labels of every non-answer token to -100."""

    def __init__(self, collator, instruction: list, response: list):
        self.collator, self.instruction, self.response = collator, instruction, response

    def __call__(self, examples):
        batch = self.collator(examples)
        ids, labels = batch["input_ids"], batch["labels"]
        positions = batch.get("position_ids")
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for row in range(ids.shape[0]):
            starts = [0]
            if positions is not None and ids.shape[0] == 1:
                starts = (positions[row] == 0).nonzero().flatten().tolist() or [0]
            bounds = starts + [ids.shape[1]]
            for begin, end in zip(bounds[:-1], bounds[1:]):
                mask = response_mask(ids[row, begin:end].tolist(), self.instruction, self.response)
                keep[row, begin:end] = torch.tensor(mask, dtype=torch.bool)
        batch["labels"] = labels.masked_fill(~keep, -100)
        return batch


def train_on_responses_only(trainer, instruction_part: str, response_part: str, **kwargs):
    """The trainer, its loss counted on the answers only.

    instruction_part and response_part are the text the chat template writes
    before a user turn and before an assistant turn, tokenized as the
    tokenizer does without special tokens of its own.
    """
    tokenizer = trainer.processing_class
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    instruction = tokenizer(instruction_part, add_special_tokens=False)["input_ids"]
    response = tokenizer(response_part, add_special_tokens=False)["input_ids"]
    if not instruction or not response:
        raise ValueError("instruction_part and response_part must each tokenize to at least one token")
    trainer.data_collator = ResponsesOnly(trainer.data_collator, instruction, response)
    return trainer
