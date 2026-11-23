"""A minimal trace viewer: send a task to the running API and see every step as a card.

    uv run python -m agent.ui            # expects the API at http://localhost:8000

Same idea as the coloured trace of the class notebook: purple for thoughts, orange for
actions, green for observations. It is your debugging tool and your demo.
"""

from __future__ import annotations

import html
import json
import os

import httpx

API_URL = os.environ.get("ARCA_API_URL", "http://localhost:8000")
COLORS = {
    "thought": ("#ede9fe", "#4c1d95"),
    "action": ("#ffedd5", "#9a3412"),
    "observation": ("#dcfce7", "#166534"),
    "final": ("#ccfbf1", "#115e59"),
}


def card(title: str, body: str, kind: str) -> str:
    bg, fg = COLORS[kind]
    return (
        f'<div style="background:{bg};color:{fg};border-left:6px solid {fg};padding:10px 14px;'
        f'margin:8px 0;border-radius:8px"><b>{html.escape(title)}</b>'
        f'<pre style="white-space:pre-wrap;margin-top:8px;color:{fg}">'
        f"{html.escape(body)}</pre></div>"
    )


def render_trace(response: dict) -> str:
    blocks = []
    for i, step in enumerate(response.get("steps", []), start=1):
        blocks.append(card(f"STEP {i} · Thought", step["thought"], "thought"))
        if step.get("action"):
            blocks.append(
                card("Action", json.dumps(step["action"], ensure_ascii=False, indent=2), "action")
            )
        if step.get("observation") is not None:
            blocks.append(card("Observation", str(step["observation"]), "observation"))
    pending = response.get("pending_actions") or []
    if pending:
        body = "\n\n".join(json.dumps(a, ensure_ascii=False, indent=2) for a in pending)
        blocks.append(card("PENDING HUMAN CONFIRMATION (not executed)", body, "action"))
    blocks.append(card("FINAL ANSWER", response.get("final_answer", ""), "final"))
    footer = f"stop_reason: {response.get('stop_reason')} · tokens: {response.get('tokens_used')}"
    blocks.append(f'<p style="color:#6b7280">{html.escape(footer)}</p>')
    return "\n".join(blocks)


def run(task: str, brain: str, max_steps: int) -> tuple[str, str, list, dict]:
    import gradio as gr

    reply = httpx.post(
        f"{API_URL}/agent", json={"task": task, "brain": brain, "max_steps": max_steps}, timeout=600
    )
    if reply.status_code != 200:
        return f"API error {reply.status_code}: {reply.text}", "", [], gr.update(visible=False)
    body = reply.json()
    pending = body.get("pending_actions") or []
    return body["final_answer"], render_trace(body), pending, gr.update(visible=bool(pending))


def confirm(pending: list) -> tuple[str, list, dict]:
    """The human approves the held actions: each one runs exactly as it was shown."""
    import gradio as gr

    cards = []
    for action in pending:
        reply = httpx.post(f"{API_URL}/agent/confirm", json=action, timeout=120)
        record = reply.json() if reply.status_code == 200 else {"ok": False, "error": reply.text}
        if record.get("ok"):
            body = json.dumps(record.get("result"), ensure_ascii=False, indent=2)
            cards.append(card(f"CONFIRMED · {action['name']} executed", body, "final"))
        else:
            error = str(record.get("error"))
            cards.append(card(f"NOT EXECUTED · {action['name']}", error, "action"))
    return "\n".join(cards), [], gr.update(visible=False)


def main() -> None:
    import gradio as gr

    with gr.Blocks(title="ARCA · agent trace") as demo:
        gr.Markdown("# ARCA · ReAct trace viewer")
        task = gr.Textbox(label="Task", lines=3)
        with gr.Row():
            # "base" by default: "rlm" does not work as a brain (see agent/RESULTADOS.md).
            brain = gr.Dropdown(["base", "rlm", "thinking"], value="base", label="Brain")
            max_steps = gr.Slider(1, 30, value=10, step=1, label="Max steps")
        button = gr.Button("Run", variant="primary")
        answer = gr.Textbox(label="Final answer", lines=4, interactive=False)
        pending = gr.State([])
        approve = gr.Button("Confirmar acciones pendientes", variant="stop", visible=False)
        confirmed = gr.HTML()
        trace = gr.HTML(label="Trace")
        button.click(
            run, inputs=[task, brain, max_steps], outputs=[answer, trace, pending, approve]
        ).then(lambda: "", outputs=confirmed)
        approve.click(confirm, inputs=pending, outputs=[confirmed, pending, approve])
    demo.launch()


if __name__ == "__main__":
    main()
