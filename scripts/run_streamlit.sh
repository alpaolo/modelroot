#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf constants/__pycache__ __pycache__
PYTHON="${ROOT}/modelroot-env/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
"$PYTHON" -c "
from constants.query import MODEL_PICKER_BROWSE_CYPHER, MODEL_PICKER_SEARCH_CYPHER, QUERY_MODULE_VERSION
print('query.py OK — version', QUERY_MODULE_VERSION)
"
exec "$PYTHON" -m streamlit run app.py "$@"
