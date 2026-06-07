#!/usr/bin/env bash
# Rebuild README.md from Tool demo/glp_tool_demo_v2.ipynb's cached outputs.
#
# Workflow:
#   1. Open Tool demo/glp_tool_demo_v2.ipynb in Jupyter.
#   2. "Restart kernel and run all cells" so every output is fresh.
#   3. Save the notebook (Cmd+S) — outputs are only persisted on disk on save.
#   4. From the repo root, run:  ./scripts/build_readme.sh
#
# Side effects:
#   - Overwrites README.md at the repo root.
#   - (Re)creates README_files/ with the PNG plot images. Track these in git.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m nbconvert "Tool demo/glp_tool_demo_v2.ipynb" \
    --to markdown \
    --no-input \
    --output README \
    --output-dir .

# Prepend the link-to-notebook banner so casual readers can jump to the
# full-code version.
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
content = p.read_text()
banner = """> 📓 **[View the full notebook (with code) →](Tool%20demo/glp_tool_demo_v2.ipynb)**
>
> This README shows the explanations and results only. Click through for the complete code, parameters, and step-by-step walkthrough.

---

"""
if "View the full notebook" not in content:
    p.write_text(banner + content)
PY

echo "Rebuilt README.md ($(wc -l < README.md) lines, $(ls README_files/ | wc -l | tr -d ' ') PNG outputs)"
