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

# Post-process the converted markdown:
#   1. Strip embedded <style>...</style> blocks (Polars DataFrame HTML +
#      the CSS-injection cell's output). GitHub strips them silently but
#      other renderers show them as literal text.
#   2. Prepend the link-to-notebook banner so casual readers can jump to
#      the full-code version.
python3 - <<'PY'
import re
from pathlib import Path

p = Path("README.md")
content = p.read_text()

# 1. Drop every <style>...</style> block and collapse blank-line runs.
content = re.sub(r"<style>.*?</style>", "", content, flags=re.DOTALL)
content = re.sub(r"\n{3,}", "\n\n", content)

# 2. Banner (idempotent).
banner = """> 📓 **[View the full notebook (with code) →](Tool%20demo/glp_tool_demo_v2.ipynb)**
>
> This README shows the explanations and results only. Click through for the complete code, parameters, and step-by-step walkthrough.

---

"""
if "View the full notebook" not in content:
    content = banner + content

# 3. Footer (idempotent).
footer = """

---

<div align="center">

## Guided Label Propagation

<img src="README_files/Q612I5.png" alt="QR code" width="220" />

**Jakob Bæk Kristensen**
RUC Digital Media Lab

</div>
"""
if "RUC Digital Media Lab" not in content:
    content = content + footer

p.write_text(content)
PY

echo "Rebuilt README.md ($(wc -l < README.md) lines, $(ls README_files/ | wc -l | tr -d ' ') PNG outputs)"
