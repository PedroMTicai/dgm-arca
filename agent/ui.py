"""The demo front end: one tab per phase, against the running API.

    uv run python -m agent.ui            # expects the API at http://localhost:8000
    ARCA_API_URL=https://... uv run python -m agent.ui    # or the public (ngrok) URL

* Fase 1 · /reasoning: our fine-tuned model codes a report in MedDRA; its reasoning, the PTs and
  the verifier's verdict.
* Fase 2 · /tools: the model picks the tools; each call with its arguments and real result.
* Fase 3 · /rag: the answer with citations and the retrieved chunks (the cited ones marked).
* Fase 4 · /agent: the ReAct trace, step by step, and the human confirmation of held actions.

Same colours as the class notebook: purple for thoughts, orange for actions, green for
observations and results. Long blocks start collapsed, with a Mostrar/Ocultar button.
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
    "error": ("#fee2e2", "#991b1b"),
}

# Examples checked against the API (docs/demo.md). One click fills the inputs.
REASONING_EXAMPLES = [
    [
        "Escribo por mi padre, de 67 años, que estaba tomando pantoprazol. Tiene antecedentes de "
        "ansiedad. Al cabo de un mes la cabeza le iba a estallar. Finalmente falleció a causa de "
        "una reacción anafiláctica.",
        "Anaphylactic reaction; Headache",
    ],
    [
        "Tengo antecedentes de asma. Llevo tres días tomando amlodipino y empecé a perder mucho "
        "pelo.",
        "Alopecia",
    ],
]
TOOLS_EXAMPLES = [
    [
        "¿La somnolencia (Somnolence) con sertralina es una señal de desproporcionalidad en "
        "FAERS? Calcula el PRR."
    ],
    [
        "Calcula el Naranjo: el mareo empezó una semana antes de que el paciente tomara el "
        "enalapril."
    ],
    ["¿Qué es la farmacovigilancia?"],
]
RAG_EXAMPLES = [
    ["¿Qué hacer ante una sobredosis de metformina?", "hybrid"],
    ["¿Con qué frecuencia produce somnolencia la sertralina?", "hybrid"],
    ["¿El omeprazol es seguro para perros?", "hybrid"],
]
AGENT_EXAMPLES = [
    [
        "Mira en la ficha técnica si el edema periférico es una reacción conocida del amlodipino "
        "y calcula su PRR en FAERS (Oedema peripheral).",
        "base",
    ],
    [
        "Registra esta notificación: una farmacéutica comunica que una mujer de 45 años que toma "
        "sertralina tuvo somnolencia, no grave, ya recuperada.",
        "base",
    ],
]


def card(title: str, body: str, kind: str) -> str:
    bg, fg = COLORS[kind]
    return (
        f'<div style="background:{bg};color:{fg};border-left:6px solid {fg};padding:10px 14px;'
        f'margin:8px 0;border-radius:8px"><b>{html.escape(title)}</b>'
        f'<pre style="white-space:pre-wrap;margin-top:8px;color:{fg}">'
        f"{html.escape(body)}</pre></div>"
    )


# Collapsed blocks: <details> folds and unfolds by itself; the CSS only swaps the button label
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
.arca-meta { color: #6b7280; }
"""


def seconds(value) -> str:
    return f"{value:.1f} s" if isinstance(value, int | float) else "—"


def fold(title: str, body: str, kind: str) -> str:
    """A block that starts collapsed: the title and a Mostrar/Ocultar button."""
    bg, fg = COLORS[kind]
    return (
        f'<details class="arca-fold" style="background:{bg};color:{fg};'
        f'border-left:6px solid {fg}"><summary>{html.escape(title)}'
        '<span class="arca-btn arca-show">Mostrar</span>'
        '<span class="arca-btn arca-hide">Ocultar</span></summary>'
        f'<pre style="color:{fg}">{html.escape(body)}</pre></details>'
    )


def meta(text: str) -> str:
    return f'<p class="arca-meta">{html.escape(text)}</p>'


def call_api(path: str, payload: dict, timeout: float = 600) -> tuple[dict | None, str]:
    """POST to the API; returns (body, "") or (None, an error card)."""
    try:
        reply = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        return None, card(f"No se pudo llamar a {path}", str(exc), "error")
    if reply.status_code != 200:
        return None, card(f"API error {reply.status_code} en {path}", reply.text, "error")
    return reply.json(), ""


# ------------------------------------------------------------------ fase 1 · /reasoning


def run_reasoning(question: str, expected: str) -> str:
    payload = {"question": question}
    if expected.strip():
        payload["expected_answer"] = expected.strip()
    body, error = call_api("/reasoning", payload)
    if body is None:
        return error
    blocks = []
    verdict = body.get("verifier")
    if verdict:
        mark = "CORRECTO" if verdict["is_correct"] else "INCORRECTO"
        detail = f"esperado: {verdict['expected']}\npredicho: {verdict['predicted']}"
        if verdict.get("detail"):
            detail += f"\n{verdict['detail']}"
        blocks.append(
            card(f"Verificador · {mark}", detail, "final" if verdict["is_correct"] else "error")
        )
    blocks.append(card("Preferred Terms (MedDRA)", body.get("answer") or "(sin <answer>)", "final"))
    words = len((body.get("thinking") or "").split())
    blocks.append(fold(f"Razonamiento ({words} palabras)", body.get("thinking") or "", "thought"))
    fmt = "formato válido" if body.get("has_valid_format") else "formato NO válido"
    blocks.append(meta(f"{fmt} · {body.get('tokens_generated')} tokens · {body.get('model')}"))
    return "\n".join(blocks)


# ---------------------------------------------------------------------- fase 2 · /tools


def run_tools(query: str) -> str:
    body, error = call_api("/tools", {"query": query})
    if body is None:
        return error
    blocks = [card("Respuesta", body.get("answer", ""), "final")]
    calls = body.get("tool_calls") or []
    if not calls:
        blocks.append(meta("No ha llamado a ninguna herramienta (no hacía falta)."))
    for call in calls:
        outcome = call.get("result") if call.get("ok") else f"error: {call.get('error')}"
        text = json.dumps(call.get("arguments", {}), ensure_ascii=False, indent=2)
        text += "\n\n— Resultado —\n" + json.dumps(outcome, ensure_ascii=False, indent=2)
        status = "" if call.get("ok") else " · ERROR"
        title = (
            f"Herramienta · {call.get('name')} ({call.get('latency_ms', 0) / 1000:.1f} s){status}"
        )
        blocks.append(fold(title, text, "action" if call.get("ok") else "error"))
    finished = "" if body.get("finished", True) else " · se agotaron los turnos"
    blocks.append(meta(f"{body.get('turns')} turnos · {body.get('model')}{finished}"))
    return "\n".join(blocks)


# ------------------------------------------------------------------------ fase 3 · /rag


def run_rag(question: str, retriever: str, top_k: int) -> str:
    body, error = call_api("/rag", {"question": question, "retriever": retriever, "top_k": top_k})
    if body is None:
        return error
    cited = set(body.get("citations") or [])
    blocks = [card("Respuesta", body.get("answer", ""), "final")]
    if cited:
        blocks.append(meta("Citas: " + ", ".join(sorted(cited))))
    if body.get("invalid_citations"):
        bad = ", ".join(body["invalid_citations"])
        blocks.append(card("Citas inventadas (no están entre los fragmentos)", bad, "error"))
    for i, chunk in enumerate(body.get("chunks") or [], start=1):
        mark = " · CITADO" if chunk["id"] in cited else ""
        title = f"{i}. [{chunk['id']}] {chunk['source']} · score {chunk['score']:.3f}{mark}"
        blocks.append(fold(title, chunk.get("text", ""), "observation"))
    blocks.append(meta(f"retriever: {body.get('retriever')} · {body.get('model')}"))
    return "\n".join(blocks)


# ---------------------------------------------------------------------- fase 4 · /agent


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
    blocks.append(meta(footer))
    return "\n".join(blocks)


def run(task: str, brain: str, max_steps: int) -> tuple[str, str, list, str, dict]:
    import gradio as gr

    body, error = call_api("/agent", {"task": task, "brain": brain, "max_steps": max_steps})
    if body is None:
        return "", error, [], "", gr.update(visible=False)
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
        body, error = call_api(
            "/agent/confirm", {"action": action, "session_id": session_id or None}, timeout=300
        )
        if body is None:
            cards.append(error)
            continue
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

    with gr.Blocks(title="ARCA · farmacovigilancia") as demo:
        gr.Markdown(f"# ARCA · agente de farmacovigilancia\nUna pestaña por fase. API: `{API_URL}`")
        with gr.Tabs():
            with gr.Tab("Fase 1 · Razonamiento"):
                gr.Markdown(
                    "Codificación MedDRA con **nuestro modelo** (Qwen3-1.7B + LoRA, SFT + GRPO). "
                    "Con la respuesta esperada, el verificador da su veredicto."
                )
                question = gr.Textbox(label="Notificación", lines=4)
                expected = gr.Textbox(label="Respuesta esperada (opcional)")
                go1 = gr.Button("Codificar", variant="primary")
                out1 = gr.HTML()
                gr.Examples(REASONING_EXAMPLES, inputs=[question, expected])
                go1.click(run_reasoning, inputs=[question, expected], outputs=out1)

            with gr.Tab("Fase 2 · Herramientas"):
                gr.Markdown(
                    "El modelo decide qué herramienta usar (FAERS, PRR, Naranjo, calculadora, "
                    "ICSR) o ninguna. Se ven las llamadas reales y sus resultados."
                )
                query = gr.Textbox(label="Pregunta", lines=3)
                go2 = gr.Button("Preguntar", variant="primary")
                out2 = gr.HTML()
                gr.Examples(TOOLS_EXAMPLES, inputs=[query])
                go2.click(run_tools, inputs=query, outputs=out2)

            with gr.Tab("Fase 3 · Conocimiento (RAG)"):
                gr.Markdown(
                    "Respuesta con citas a las fichas técnicas de CIMA y las guías de la EMA, o "
                    "«no lo sé» si no está en el corpus."
                )
                rag_q = gr.Textbox(label="Pregunta", lines=2)
                with gr.Row():
                    retriever = gr.Dropdown(
                        ["hybrid", "dense", "bm25"], value="hybrid", label="Retriever"
                    )
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Fragmentos")
                go3 = gr.Button("Preguntar", variant="primary")
                out3 = gr.HTML()
                gr.Examples(RAG_EXAMPLES, inputs=[rag_q, retriever])
                go3.click(run_rag, inputs=[rag_q, retriever, top_k], outputs=out3)

            with gr.Tab("Fase 4 · Agente"):
                gr.Markdown(
                    "El bucle ReAct con todas las herramientas. Las acciones con efecto esperan "
                    "confirmación humana."
                )
                task = gr.Textbox(label="Tarea", lines=3)
                with gr.Row():
                    # "base" here: the fastest for a live demo; "rlm" does not work as a brain
                    # (agent/RESULTADOS.md); the API default is "thinking".
                    brain = gr.Dropdown(["base", "thinking", "rlm"], value="base", label="Cerebro")
                    max_steps = gr.Slider(1, 30, value=10, step=1, label="Pasos máximos")
                button = gr.Button("Ejecutar", variant="primary")
                answer = gr.Textbox(label="Respuesta final", lines=4, interactive=False)
                pending = gr.State([])
                session = gr.State("")
                approve = gr.Button("Confirmar acciones pendientes", variant="stop", visible=False)
                confirmed = gr.HTML()
                trace = gr.HTML(label="Traza")
                gr.Examples(AGENT_EXAMPLES, inputs=[task, brain])
                button.click(
                    run,
                    inputs=[task, brain, max_steps],
                    outputs=[answer, trace, pending, session, approve],
                ).then(lambda: "", outputs=confirmed)
                approve.click(
                    confirm, inputs=[pending, session], outputs=[confirmed, pending, approve]
                )
    demo.launch(css=CSS)  # Gradio 6 takes custom CSS in launch()


if __name__ == "__main__":
    main()
