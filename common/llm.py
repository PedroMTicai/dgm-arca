"""One chat model shared by phases 2, 3 and 4.

The API serves every phase from a single process on a 16 GB GPU, so the generator of
``/rag``, the model behind ``/tools`` and the "base" brain of ``/agent`` are the same
instance: ``get_chat_model`` caches one ``ChatModel`` per (model id, adapter) pair.

Configuration: ``ARCA_LLM_MODEL`` (default ``Qwen/Qwen3-1.7B``). Qwen3 is a hybrid model:
``enable_thinking=False`` makes it answer directly, which is what RAG wants (the reasoning
is in the retrieved text, not in a long chain of thought); the agent can turn it on.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

DEFAULT_LLM = "Qwen/Qwen3-1.7B"
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a ``<think>…</think>`` block (and an unterminated one) from a completion."""
    text = _THINK.sub("", text)
    return text.split("<think>")[0].strip() if "<think>" in text else text.strip()


def cut_at_stop(text: str, stop: list[str] | None) -> str:
    """Text up to the first stop string, ignoring stop strings inside a ``<think>`` block."""
    if not stop:
        return text
    start = text.find("</think>")
    if start >= 0:
        start += len("</think>")
    elif "<think>" in text:
        return text  # still thinking
    else:
        start = 0
    cuts = [i for s in stop if (i := text.find(s, start)) >= 0]
    return text[: min(cuts)] if cuts else text


class ChatModel:
    """A Hugging Face causal LM (plus optional LoRA adapter) behind a chat interface."""

    def __init__(self, model_id: str = DEFAULT_LLM, adapter: str | None = None):
        self.model_id = model_id
        self.adapter = adapter
        self.model = None
        self.tokenizer = None
        self.last_prompt_tokens = 0
        self._lock = threading.Lock()  # one generation at a time on the shared GPU

    @property
    def name(self) -> str:
        return (
            f"{self.model_id}+{os.path.basename(self.adapter)}" if self.adapter else self.model_id
        )

    def load(self) -> ChatModel:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        self.model = model.eval()
        return self

    def count_tokens(self, text: str) -> int:
        """Tokens of a text with this model's tokenizer (the agent's memory budget)."""
        if self.tokenizer is None:
            self.load()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """The prompt string the model actually sees (useful to show in the README)."""
        kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        """One completion for the conversation. Returns ``(raw text, generated tokens)``.

        ``temperature=0`` is greedy decoding, so answers are reproducible for the grader.
        ``stop`` ends the generation at the first of those strings, which is cut from the
        text (the code agent stops at "Observation:" so the model does not invent one).
        """
        import torch

        if self.model is None:
            self.load()
        prompt = self.render(messages, tools=tools, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])  # cost accounting (phase 4)
        sampling = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            if temperature > 0
            else {"do_sample": False}
        )
        if stop:
            # Not transformers' stop_strings: those would also fire inside <think>.
            sampling["stopping_criteria"] = [self._stop_criterion(stop, inputs)]
        with self._lock, torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        # Keep <think>/<tool_call> markers (they are not special tokens in Qwen3) but drop
        # <|im_end|> and padding.
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return cut_at_stop(text, stop), int(new_tokens.shape[0])

    def _stop_criterion(self, stop: list[str], inputs):
        """Stop once a stop string appears after the thinking block (one sequence)."""
        import torch
        from transformers import StoppingCriteria

        tokenizer, prompt_len = self.tokenizer, int(inputs["input_ids"].shape[1])

        class StopAfterThinking(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                done = cut_at_stop(text, stop) != text
                return torch.full((input_ids.shape[0],), done, device=input_ids.device)

        return StopAfterThinking()


_MODELS: dict[tuple[str, str | None], ChatModel] = {}
_MODELS_LOCK = threading.Lock()


def get_chat_model(model_id: str | None = None, adapter: str | None = None) -> ChatModel:
    """The process-wide instance for ``(model_id, adapter)``, loaded on first use."""
    key = (model_id or os.environ.get("ARCA_LLM_MODEL") or DEFAULT_LLM, adapter)
    with _MODELS_LOCK:
        if key not in _MODELS:
            _MODELS[key] = ChatModel(*key).load()
        return _MODELS[key]
