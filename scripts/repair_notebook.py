import json
import os
import sys

nb_path = os.path.abspath("notebooks/MathSLM_v2_diagnostics.ipynb")

print(f"Reading {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    raw_content = f.read()

# Identify the exact issue
target = '"    print(\'Backing up checkpoints and logs to Google Drive...\')\n'
replacement = '"    print(\'Backing up checkpoints and logs to Google Drive...\')\\n",\n'

if target in raw_content:
    print("Found unescaped control character newline at target location!")
else:
    print("WARNING: target literal string not found directly, checking regex / position.")

fixed_content = raw_content.replace(target, replacement)

# Validate JSON parsing
try:
    nb_dict = json.loads(fixed_content)
    print("✅ JSON validation successful!")
except Exception as e:
    print(f"❌ JSON validation failed: {e}")
    sys.exit(1)

# Write out the repaired JSON formatted cleanly via json.dump
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Saved repaired notebook to {nb_path}")

# Re-validate with json
with open(nb_path, "r", encoding="utf-8") as f:
    reloaded_dict = json.load(f)
print("✅ Re-loaded repaired notebook JSON successfully.")

# Check nbformat structure
assert "nbformat" in reloaded_dict, "Missing nbformat"
assert "nbformat_minor" in reloaded_dict, "Missing nbformat_minor"
assert "cells" in reloaded_dict, "Missing cells"

cells = reloaded_dict["cells"]
print(f"Total notebook cells: {len(cells)}")

for idx, cell in enumerate(cells, 1):
    cell_type = cell.get("cell_type")
    assert cell_type in ["code", "markdown", "raw"], f"Invalid cell type in cell {idx}: {cell_type}"
    source = cell.get("source")
    assert source is not None, f"Missing source in cell {idx}"
    src_text = "".join(source) if isinstance(source, list) else source
    first_line = src_text.strip().split("\n")[0] if src_text.strip() else "(empty)"
    print(f"Cell {idx} [{cell_type}]: {first_line}")

# Validate with nbformat if installed
try:
    import nbformat
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    print("✅ Jupyter nbformat library validation PASSED!")
except ImportError:
    print("nbformat library not installed, JSON validation passed.")
except Exception as e:
    print(f"nbformat validation warning: {e}")
