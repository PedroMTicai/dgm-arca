"""Phase 1, step 2: cold-start supervised fine-tuning on verified reasoning traces.

This is the "SFT round 1" box of the DeepSeek pipeline. The input is the JSONL produced
by ``rlm/distill.py``: problems from your domain with a teacher-generated reasoning trace
that *passed your verifier*. The output is a LoRA adapter that already speaks the
``<think>…</think><answer>…</answer>`` format before we ever run GRPO.

Run::

    uv run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --output rlm/weights/sft_lora

What is decided for you: the trainer (``trl.SFTTrainer``), LoRA, the conversational
format (``prompt`` + ``completion`` columns, so that the loss is computed only on the
assistant turn). What you decide: the data, the base model, epochs, learning rate,
sequence length. Write down in EXPERIMENTS.md why you chose them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from rlm.data import build_prompt


def load_sft_dataset(path: str | Path):
    """Turn distilled traces into TRL's prompt/completion conversational format.

    Expected JSONL fields: ``question``, ``trace`` (the full ``<think>…</think><answer>…</answer>``
    text) and ``answer``. Only traces marked ``verified: true`` are used.
    """
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "verified" in dataset.column_names:
        dataset = dataset.filter(lambda ex: bool(ex["verified"]))
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"]),
            "completion": [{"role": "assistant", "content": ex["trace"]}],
        },
        remove_columns=dataset.column_names,
    )


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_sft_dataset(args.data)
    print(f"{len(dataset)} verified traces loaded from {args.data}")
    eval_dataset = None
    if args.eval_fraction > 0:
        split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        dataset, eval_dataset = split["train"], split["test"]
        print(f"{len(dataset)} for training, {len(eval_dataset)} held out for eval loss")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        # A rising eval loss while the train loss keeps falling means the adapter is
        # memorising the traces: stop earlier or lower the epochs.
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        # No packing: our traces are short (a few hundred tokens) and packing would mix
        # unrelated reports in one sequence. The prompt/completion format already restricts
        # the loss to the assistant turn, so `assistant_only_loss` is not needed.
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    trainer.save_state()  # trainer_state.json: loss curves for `rlm.evaluate --history`
    print(f"adapter and trainer_state.json saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", required=True, help="JSONL with question / trace / answer / verified"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="rlm/weights/sft_lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.05, help="0 disables eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None, help="checkpoint-XXX folder")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
