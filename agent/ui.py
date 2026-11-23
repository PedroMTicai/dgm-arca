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


# Collapsed steps: <details> folds and unfolds by itself; the CSS only swaps the button label
# ("Mostrar" while closed, "Ocultar" while open). Passed to demo.launch(css=...).
CSS = """
.arca-fold { border-radius: 8px; margin: 6px 0; padding: 6px 12px; }
.arca-fold > summary { cursor: pointer; list-style: none; display: flex; gap: 10px;
                       align-items: center; font-weight: 600; }
.arca-fold > summary::-webkit-details-marker { display: none; }
.arca-fold .arca-btn { margin-left: auto; font-size: 0.8em; font-weight: 500; padding: 2px 10px;
                       border-radius: 6px; border: 1px solid currentColor; }
.arca-fold .arca-hide, .arca-fold[open] .arca-show { display: none; }
.arca-fold[open] .arca-hide { display: inline; }
.arca-fold pre { white-space: pre-wrap; margin: 8px 0 4px; }
"""


def seconds(value) -> str:
    return f"{value:.1f} s" if isinstance(value, int | float) else "—"


def fold(title: str, body: str, kind: str) -> str:
    """A step that starts collapsed: the title and a Mostrar/Ocultar button."""
    bg, fg = COLORS[kind]
    return (
        f'<details class="arca-fold" style="background:{bg};color:{fg};'
        f'border-left:6px solid {fg}"><summary>{html.escape(title)}'
        '<span class="arca-btn arca-show">Mostrar</span>'
        '<span class="arca-btn arca-hide">Ocultar</span></summary>'
        f'<pre style="color:{fg}">{html.escape(body)}</pre></details>'
    )


def render_trace(response: dict) -> str:
    blocks = []
    for i, step in enumerate(response.get("steps", []), start=1):
        title = f"Paso {i} · Thought for {seconds(step.get('thought_seconds'))}"
        blocks.append(fold(title, step["thought"], "thought"))
        action = step.get("action")
        if action:
            body = json.dumps(action.get("arguments", {}), ensure_ascii=False, indent=2)
            if step.get("observation") is not None:
                body += "\n\n— Observation —\n" + str(step["observation"])
            name = action.get("name", "")
            if name == "final_answer":
                continue  # its text is the final answer card below
            title = f"Action · {name} ({seconds(step.get('action_seconds'))})"
            blocks.append(fold(title, body, "action"))
        elif step.get("observation") is not None:
            blocks.append(fold("Observation", str(step["observation"]), "observation"))
    pending = response.get("pending_actions") or []
    if pending:
        body = "\n\n".join(json.dumps(a, ensure_ascii=False, indent=2) for a in pending)
        blocks.append(card("PENDING HUMAN CONFIRMATION (not executed)", body, "action"))
    blocks.append(card("FINAL ANSWER", response.get("final_answer", ""), "final"))
    footer = f"stop_reason: {response.get('stop_reason')} · tokens: {response.get('tokens_used')}"
    blocks.append(f'<p style="color:#6b7280">{html.escape(footer)}</p>')
    return "\n".join(blocks)


def run(task: str, brain: str, max_steps: int) -> tuple[str, str, list, str, dict]:
    import gradio as gr

    reply = httpx.post(
        f"{API_URL}/agent", json={"task": task, "brain": brain, "max_steps": max_steps}, timeout=600
    )
    if reply.status_code != 200:
        error = f"API error {reply.status_code}: {reply.text}"
        return error, "", [], "", gr.update(visible=False)
    body = reply.json()
    pending = body.get("pending_actions") or []
    return (
        body["final_answer"],
        render_trace(body),
        pending,
        body.get("session_id") or "",
        gr.update(visible=bool(pending)),
    )


def confirm(pending: list, session_id: str) -> tuple[str, list, dict]:
    """The human approves the held actions: each one runs exactly as it was shown, and the
    agent (same conversation) tells the user the outcome."""
    import gradio as gr

    cards = []
    for action in pending:
        reply = httpx.post(
            f"{API_URL}/agent/confirm",
            json={"action": action, "session_id": session_id or None},
            timeout=300,
        )
        if reply.status_code != 200:
            cards.append(card(f"NOT EXECUTED · {action['name']}", reply.text, "action"))
            continue
        body = reply.json()
        record = body["record"]
        cards.append(card("AGENT · after confirmation", body["message"], "thought"))
        if record.get("ok"):
            result = json.dumps(record.get("result"), ensure_ascii=False, indent=2)
            cards.append(card(f"CONFIRMED · {action['name']} executed", result, "final"))
        else:
            cards.append(card(f"NOT EXECUTED · {action['name']}", str(record["error"]), "action"))
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
        session = gr.State("")
        approve = gr.Button("Confirmar acciones pendientes", variant="stop", visible=False)
        confirmed = gr.HTML()
        trace = gr.HTML(label="Trace")
        button.click(
            run,
            inputs=[task, brain, max_steps],
            outputs=[answer, trace, pending, session, approve],
        ).then(lambda: "", outputs=confirmed)
        approve.click(confirm, inputs=[pending, session], outputs=[confirmed, pending, approve])
    demo.launch(css=CSS)  # Gradio 6 takes custom CSS in launch()


if __name__ == "__main__":
    main()
