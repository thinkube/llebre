# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from llebre import LlebreLanguageModel
from llebre.chat_templates import response_mask, train_on_responses_only

U, A = [1], [2]  # instruction part, response part


def test_the_mask_covers_each_answer_and_nothing_else():
    ids = [9, 1, 5, 5, 2, 7, 7, 1, 5, 2, 8]
    assert response_mask(ids, U, A) == [False, False, False, False, False, True, True, False, False, False, True]


def test_no_response_part_masks_everything():
    assert response_mask([9, 1, 5, 5], U, A) == [False] * 4


TALK = Dataset.from_dict({"text": ["the user runs and the hare fast", "user sloth hare sleeps"]})


@pytest.fixture
def words(tokenizer):
    """The test tokenizer, with "user" and "hare" standing for the instruction and response parts."""
    tokenizer.add_tokens(["user"])
    return tokenizer


@pytest.mark.parametrize("padding_free", [True, False])
def test_only_answer_tokens_keep_their_labels(tiny_model, words, tmp_path, padding_free):
    tiny_model.resize_token_embeddings(len(words))
    model = LlebreLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1,
                     padding_free=padding_free, max_length=None)
    trainer = train_on_responses_only(SFTTrainer(model=model, args=args, train_dataset=TALK, processing_class=words),
                                      instruction_part="user", response_part="hare")
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
    kept = [words.convert_ids_to_tokens(int(i)) for i, label in zip(batch["input_ids"].flatten(), batch["labels"].flatten())
            if label != -100]
    # "fast" after the first example's "hare", "sleeps" after the second's (and its end-of-sequence)
    assert [t for t in kept if t != words.eos_token] == ["fast", "sleeps"]
