#!/usr/bin/env bash
# Sirve la API de ARCA desde una sesión de la DGX y la publica con ngrok para la corrección.
#
#   cd ~/dgm-arca && git pull
#   export NGROK_AUTHTOKEN=...          # de https://dashboard.ngrok.com (no lo guardéis en el repo)
#   export NGROK_DOMAIN=xxx.ngrok-free.app   # opcional: el dominio fijo gratuito de la cuenta,
#                                            # para que la URL no cambie entre sesiones
#   bash scripts/dgx_serve.sh           # levanta todo y escribe la URL pública
#   bash scripts/dgx_serve.sh stop      # lo para
#
# Qué hace: prepara el entorno (dgx_env.sh), crea .env si falta, comprueba el corpus, arranca la
# API en segundo plano, la calienta cargando las cuatro fases (/health), descarga ngrok si no está
# y abre el túnel. Logs en $OUT. La sesión de la DGX dura unas horas: al cerrarse, la URL deja de
# responder y hay que volver a lanzar el script en una sesión nueva.
set -u

WORK=${ARCA_WORK_DIR:-$HOME/clusters/dgx}
OUT=$WORK/arca_serve
BIN=$WORK/bin
# A port of our own: the DGX is shared and every team runs the same ARCA template on 8000, so
# "something answers on 8000" may be another team's API. 20000 + uid % 10000 is fixed per user.
PORT=${ARCA_SERVE_PORT:-$((20000 + $(id -u) % 10000))}
export ARCA_API_PORT=$PORT
ME=$(id -un)   # not $USER: it is not always set in the DGX terminals
mkdir -p "$OUT" "$BIN"

if [[ "${1:-}" == "stop" ]]; then
    # -u $ME: the DGX is shared, never touch other people's processes.
    pkill -u "$ME" -f "ngrok http" && echo "ngrok parado" || echo "ngrok no estaba en marcha"
    pkill -u "$ME" -f "arca-api" && echo "API parada" || echo "la API no estaba en marcha"
    rm -f "$OUT/api.pid"
    exit 0
fi

# Cachés de uv y de Hugging Face en la zona persistente (no se descargan otra vez cada sesión).
# shellcheck disable=SC1091
source smoke/dgx_env.sh || exit 1

[[ -f .env ]] || { cp .env.example .env; echo "Creado .env a partir de .env.example"; }
uv sync --extra train --extra rag --extra agent || exit 1

# El corpus no está en git. download_corpus solo baja lo que falta; si la EMA rechaza un PDF,
# avisa y sigue, así que se comprueba aquí que estén las tres guías.
uv run python -m rag.download_corpus > "$OUT/corpus.log" 2>&1
n_guides=$(ls rag/corpus/guidelines/*.pdf 2>/dev/null | wc -l)
if (( n_guides < 3 )); then
    echo "!!! Solo hay $n_guides de 3 guías de la EMA en rag/corpus/guidelines (ver $OUT/corpus.log)."
    echo "    Vuelve a lanzar el script o cópialas desde otra máquina antes de la corrección."
fi

# 1. La API, en segundo plano, en nuestro puerto. Solo se reutiliza si es la que lanzamos nosotros.
ours=false
[[ -f $OUT/api.pid ]] && kill -0 "$(cat "$OUT/api.pid")" 2>/dev/null && ours=true
if ! $ours; then
    if curl -s -o /dev/null "localhost:$PORT/docs"; then
        echo "El puerto $PORT está ocupado por otro proceso. Elige otro: ARCA_SERVE_PORT=2xxxx bash $0"
        exit 1
    fi
    nohup uv run arca-api > "$OUT/api.log" 2>&1 &
    echo $! > "$OUT/api.pid"
    echo "API arrancando en el puerto $PORT (log: $OUT/api.log)…"
    for _ in $(seq 1 90); do curl -s -o /dev/null "localhost:$PORT/docs" && break; sleep 2; done
fi
curl -s -o /dev/null "localhost:$PORT/docs" || { echo "La API no arranca: mira $OUT/api.log"; exit 1; }

# 2. Calentarla: /health carga las cuatro fases, para que la primera petición del corrector no
# tarde minutos.
echo "Cargando las cuatro fases (unos minutos)…"
curl -s -m 1800 "localhost:$PORT/health" > "$OUT/health.json"
python3 - "$OUT/health.json" <<'EOF'
import json, sys
health = json.load(open(sys.argv[1]))
print(f"  equipo: {health.get('team') or '(ARCA_TEAM vacío en .env)'}")
for name, status in health["phases"].items():
    print(f"  {name:10s} {status['status']}  {status.get('detail') or ''}")
EOF

# 3. ngrok: un único ejecutable, sin permisos de administrador.
if [[ ! -x $BIN/ngrok ]]; then
    case $(uname -m) in
        x86_64) arch=amd64 ;;
        aarch64) arch=arm64 ;;
        *) echo "Arquitectura no prevista: $(uname -m)"; exit 1 ;;
    esac
    curl -sSL "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-$arch.tgz" | tar -xz -C "$BIN"
fi
if [[ -z ${NGROK_AUTHTOKEN:-} ]]; then
    echo "Falta NGROK_AUTHTOKEN (export NGROK_AUTHTOKEN=... con el token de tu cuenta de ngrok)."
    exit 1
fi
pkill -u "$ME" -f "ngrok http" 2>/dev/null
sleep 2  # ngrok needs a moment to release the endpoint on its servers
args=(http "$PORT" --log stdout)
[[ -n ${NGROK_DOMAIN:-} ]] && args+=(--url "$NGROK_DOMAIN")
nohup "$BIN/ngrok" "${args[@]}" > "$OUT/ngrok.log" 2>&1 &

# The URL is read from OUR ngrok log, not from its local panel (localhost:4040): the DGX is
# shared, the panel port may belong to someone else's ngrok and give their URL.
url=""
for _ in $(seq 1 30); do
    url=$(grep -o 'url=https://[^ ]*' "$OUT/ngrok.log" 2>/dev/null | tail -n 1 | cut -d= -f2)
    [[ -n $url ]] && break
    sleep 1
done
if [[ -z $url ]]; then
    echo "ngrok no ha abierto el túnel ($OUT/ngrok.log):"
    grep "^ERROR" "$OUT/ngrok.log" | head -n 3
    echo "Si dice que el endpoint ya está online: bash scripts/dgx_serve.sh stop, y vuelve a lanzarlo."
    exit 1
fi
echo "$url" > "$OUT/url.txt"
echo
echo "API pública: $url        (documentación: $url/docs)"
echo "Comprobación: curl -s $url/health"
