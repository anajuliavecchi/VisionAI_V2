
"""Integra o histórico persistente ao api_server.py existente.

Execute este script na pasta em que está o api_server.py.
Ele cria api_server.py.bak antes de alterar o arquivo.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "api_server.py"

if not TARGET.exists():
    TARGET = Path.cwd() / "api_server.py"

if not TARGET.exists():
    raise SystemExit(
        "ERRO: api_server.py não encontrado. Copie este script e "
        "event_store.py para a pasta do servidor."
    )

source = TARGET.read_text(encoding="utf-8")
original = source

if "from event_store import EventStore" not in source:
    match = re.search(
        r"^from fastapi import ([^\n]+)$",
        source,
        re.M,
    )

    if not match:
        raise SystemExit(
            "ERRO: não encontrei 'from fastapi import ...'. "
            "Nenhum arquivo foi alterado."
        )

    names = [
        part.strip()
        for part in match.group(1).split(",")
    ]

    if "Query" not in names:
        names.append("Query")

    source = (
        source[:match.start()]
        + "from fastapi import " + ", ".join(names)
        + source[match.end():]
    )

    response_import = re.search(
        r"^from fastapi\.responses import StreamingResponse\s*$",
        source,
        re.M,
    )

    if response_import:
        pos = response_import.end()
        source = (
            source[:pos]
            + "\n\nfrom event_store import EventStore"
            + source[pos:]
        )
    else:
        source = "from event_store import EventStore\n" + source

if "event_store = EventStore(" not in source:
    root_anchor = re.search(
        r"sys\.path\.insert\(\s*0\s*,\s*ROOT_DIR\s*\)",
        source,
    )

    if not root_anchor:
        raise SystemExit(
            "ERRO: não encontrei a configuração ROOT_DIR/sys.path.insert. "
            "Nenhum arquivo foi alterado."
        )

    insertion = """
    
# Histórico persistente de incidentes do dashboard.
event_store = EventStore(
    db_path=os.path.join(ROOT_DIR, "data", "visionai_events.sqlite3"),
    confirm_seconds=4.0,
    resolve_seconds=3.0,
)
"""
    source = (
        source[:root_anchor.end()]
        + insertion
        + source[root_anchor.end():]
    )

if "event_store.observe(detected_faces)" not in source:
    observe_anchor = re.search(
        r"^(\s*)detected_faces\s*=\s*temporal_filter\.update\(detected_faces\)\s*$",
        source,
        re.M,
    )

    if not observe_anchor:
        raise SystemExit(
            "ERRO: não encontrei a atualização do temporal_filter. "
            "Nenhum arquivo foi alterado."
        )

    indent = observe_anchor.group(1)
    insertion = (
        "\n" + indent + "try:\n"
        + indent + "    event_store.observe(detected_faces)\n"
        + indent + "except Exception as exc:\n"
        + indent + "    # Histórico não deve interromper a câmera/inferência.\n"
        + indent + '    print(f"[EVENTS] Falha ao registrar evento: {exc}")'
    )

    source = (
        source[:observe_anchor.end()]
        + insertion
        + source[observe_anchor.end():]
    )

if (
    '@app.get("/api/events")' not in source
    and '@app.get(\n    "/api/events"' not in source
):
    startup = re.search(
        r"@app\.on_event\(\s*[\"']startup[\"']\s*\)",
        source,
    )

    if not startup:
        raise SystemExit(
            "ERRO: não encontrei o evento startup da FastAPI. "
            "Nenhum arquivo foi alterado."
        )

    endpoints = '''@app.get("/api/events")
def api_events(
    limit: int = Query(default=200, ge=1, le=1000),
    status: str = Query(default="all", regex="^(all|active|resolved)$"),
):
    return {
        "events": event_store.list_events(limit=limit, status=status),
        "summary": event_store.summary(),
    }


@app.get("/api/events/summary")
def api_events_summary():
    return event_store.summary()


'''

    source = source[:startup.start()] + endpoints + source[startup.start():]

if source == original:
    print("OK: a integração já estava aplicada.")
else:
    backup = TARGET.with_suffix(TARGET.suffix + ".bak")

    if not backup.exists():
        shutil.copy2(TARGET, backup)

    TARGET.write_text(source, encoding="utf-8")

    print(f"OK: integração aplicada em {TARGET}")
    print(f"Backup: {backup}")
    print("Reinicie a API e teste /api/events e /api/events/summary.")
