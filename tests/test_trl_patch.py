# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from llebre import FastLanguageModel

TEXT = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps"]})


def trainer(model, tokenizer, tmp_path, **config):
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1, **config)
    return SFTTrainer(model=model, args=args, train_dataset=TEXT, processing_class=tokenizer)


def test_padding_free_and_checkpointing_come_from_the_model(tiny_model, tokenizer, tmp_path):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.padding_free is True
    assert t.args.gradient_checkpointing is False
    assert t.args.use_liger_kernel is False


def test_a_model_that_cannot_go_padding_free_is_padded(tiny_model, tokenizer, tmp_path):
    tiny_model._llebre = {"padding_free": False, "liger": False}
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.padding_free is False
    assert t.args.gradient_checkpointing is True


def test_the_config_wins_when_it_sets_them(tiny_model, tokenizer, tmp_path):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    t = trainer(model, tokenizer, tmp_path, padding_free=False, gradient_checkpointing=False)
    assert t.args.padding_free is False
    assert not model.base_model.model.is_gradient_checkpointing


def test_other_models_are_left_alone(tiny_model, tokenizer, tmp_path):
    del tiny_model._llebre
    t = trainer(tiny_model, tokenizer, tmp_path, gradient_checkpointing=False)
    assert t.args.padding_free is False


def test_examples_are_cut_to_max_length(tiny_model, tokenizer, tmp_path):
    long = Dataset.from_dict({"text": ["the hare runs fast and the sloth sleeps"] * 2})
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1, max_length=3)
    t = SFTTrainer(model=model, args=args, train_dataset=long, processing_class=tokenizer)
    assert t.args.padding_free is True and t.args.max_length == 3
    batch = t.data_collator([t.train_dataset[0], t.train_dataset[1]])
    # padding-free: the two examples end to end in one row, each cut to 3 tokens
    assert batch["input_ids"].shape == (1, 6)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]


def test_grpo_takes_its_settings_from_the_model(tiny_model, tokenizer, tmp_path):
    from trl import GRPOConfig, GRPOTrainer

    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    prompts = Dataset.from_dict({"prompt": ["the hare", "the sloth"]})
    args = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1,
                      per_device_train_batch_size=2, num_generations=2, max_completion_length=4)
    # reward_funcs by position, as GRPOTrainer's signature puts it before args
    t = GRPOTrainer(model, lambda completions, **kw: [0.0] * len(completions), args=args,
                    train_dataset=prompts, processing_class=tokenizer)
    assert t.args.gradient_checkpointing is True
    assert t.args.use_liger_kernel is False
