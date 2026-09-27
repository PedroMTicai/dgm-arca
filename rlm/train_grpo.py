"""Phase 1, step 3: reinforcement learning with verifiable rewards using GRPO.

Starting point: the SFT adapter from ``train_sft.py`` (or the base model, if you want to
reproduce the R1-Zero experiment and see what happens without cold start). Output: your
final reasoning model, ``rlm/weights/final_rlm_lora``.

Run::

    uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --init-adapter rlm/weights/sft_lora

Rewards for our MedDRA coding task (``--verifier meddra``, the default for a JSONL dataset):

* ``format_reward``: the ``<think>…</think><answer>…</answer>`` structure.
* ``accuracy_reward``: ``MedDRAVerifier`` on the answer, 1.0 if the set of Preferred Terms is
  exactly the expected one. With ``--partial-credit`` it is the F1 between both sets instead,
  a denser signal for the experiment on sparse rewards.
* ``domain_reward``: MedDRA vocabulary validity (``rlm.meddra.validity_score``). An E2B
  report with a term that is not in MedDRA is rejected by the regulator, and a term at the
  wrong level (an LLT, a US spelling) needs a human to fix it. So each name in the answer
  scores 1.0 if it is an exact PT, 0.5 if it is an accepted alternative and 0.0 otherwise,
  and enumerating more than five terms scores 0.0 so that listing valid PTs is not a shortcut.

The smoke test (``smoke/smoke_grpo.py``) is the minimal version of this script on GSM8K.
Here you add what makes it *yours*:

1. Your domain dataset (``rlm/data.py::load_domain_dataset``) and your verifier.
2. A third, domain-specific reward (``domain_reward`` below). Think about what a good
   answer looks like for your user beyond being correct: language, units, length,
   citing a source, respecting a schema. Justify it in EXPERIMENTS.md and, if you use
   ``reward_weights``, justify those too.
3. The hyper-parameters. Group size, completion length, learning rate and KL
   coefficient all change what the model learns. Change one thing at a time and log it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.meddra import MedDRADictionary, term_f1, validity_score
from rlm.rewards import _completion_text, accuracy_reward, extract_answer, format_reward
from rlm.verifier import MedDRAVerifier, Verifier, build_verifier

_DICTIONARY: MedDRADictionary | None = None


def _dictionary() -> MedDRADictionary:
    global _DICTIONARY
    if _DICTIONARY is None:
        _DICTIONARY = MedDRADictionary.default()
    return _DICTIONARY


def domain_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """MedDRA vocabulary validity of the answer, in [0, 1] (see the module docstring).

    It does not look at the ground truth on purpose: it rewards *codable* answers, while
    ``accuracy_reward`` rewards *correct* ones. A completion without an ``<answer>`` block
    scores 0.0, which the format reward already punishes too.
    """
    dictionary = _dictionary()
    return [validity_score(extract_answer(_completion_text(c)), dictionary) for c in completions]


def make_accuracy_reward(verifier: Verifier, partial_credit: bool = False):
    """Accuracy reward driven by a verifier, with the signature ``GRPOTrainer`` expects.

    With ``partial_credit`` (MedDRA only) the reward is the F1 between predicted and expected
    PT sets instead of 0/1 set equality.
    """

    def accuracy_reward(
        prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
    ) -> list[float]:
        rewards = []
        for completion, expected in zip(completions, answer, strict=True):
            text = _completion_text(completion)
            if partial_credit and isinstance(verifier, MedDRAVerifier):
                rewards.append(term_f1(extract_answer(text), expected, verifier.dictionary))
            else:
                rewards.append(1.0 if verifier.verify(text, expected).is_correct else 0.0)
        return rewards

    return accuracy_reward


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.data == "gsm8k":
        dataset = load_gsm8k("train", n_examples=args.n_examples, seed=args.seed)
        reward_funcs = [format_reward, accuracy_reward]
    else:
        dataset = load_domain_dataset(args.data)
        if args.n_examples:
            dataset = dataset.shuffle(seed=args.seed).select(
                range(min(args.n_examples, len(dataset)))
            )
        verifier = build_verifier(args.verifier)
        reward_funcs = [format_reward, make_accuracy_reward(verifier, args.partial_credit)]
        if args.verifier == "meddra":
            reward_funcs.append(domain_reward)
    weights = args.reward_weights[: len(reward_funcs)]
    print(
        f"{len(dataset)} training problems; rewards: "
        + ", ".join(f"{f.__name__}×{w}" for f, w in zip(reward_funcs, weights, strict=True))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=0.2,
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        reward_weights=weights,
    )

    if args.init_adapter:
        # Continue training the SFT adapter: load base + adapter as a trainable PeftModel.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    # trainer_state.json holds log_history: rewards, KL, completion length per step.
    # `rlm.evaluate --history` turns it into the training curves.
    trainer.save_state()
    print(f"final adapter and trainer_state.json saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' or path to your domain JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter", default=None, help="SFT adapter to start from")
    parser.add_argument("--output", default="rlm/weights/final_rlm_lora")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 disables it)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="path to a checkpoint-XXX folder to continue an interrupted run (24h sessions!)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verifier", default="meddra", help="verifier for a JSONL dataset (see rlm/verifier.py)"
    )
    parser.add_argument(
        "--partial-credit",
        action="store_true",
        help="accuracy = F1 between PT sets instead of exact set match (MedDRA only)",
    )
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs="+",
        default=[0.5, 2.0, 0.5],
        help="weights for format / accuracy / domain rewards",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
