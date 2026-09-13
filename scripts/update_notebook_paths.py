import json
import os
import sys

nb_path = "notebooks/MathSLM_v2_diagnostics.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_2_code = [
    "# Step 1: Mount Google Drive & Setup Repository Working Directory\n",
    "import os, sys\n",
    "\n",
    "# 1. Mount Google Drive (if in Colab)\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    DRIVE_DIR = '/content/drive/MyDrive/MathSLM_v2'\n",
    "    os.makedirs(DRIVE_DIR, exist_ok=True)\n",
    "    print('Google Drive mounted successfully at', DRIVE_DIR)\n",
    "except Exception as e:\n",
    "    print('Running locally or Google Drive not mounted:', e)\n",
    "\n",
    "# 2. Auto-detect and set working directory to repository root\n",
    "repo_name = 'KLH-csit-2026-2420090008-mathSlm'\n",
    "target_path = f'/content/{repo_name}'\n",
    "\n",
    "if os.path.exists(target_path):\n",
    "    os.chdir(target_path)\n",
    "elif not os.path.exists('data/generate_synthetic_v2.py') and os.path.exists('/content'):\n",
    "    print(f'Cloning {repo_name} repository...')\n",
    "    !git clone https://github.com/Ankitt-02/KLH-csit-2026-2420090008-mathSlm.git\n",
    "    if os.path.exists(target_path):\n",
    "        os.chdir(target_path)\n",
    "\n",
    "print('✅ Current Working Directory:', os.getcwd())\n",
    "if os.path.exists('data/generate_synthetic_v2.py'):\n",
    "    print('✅ Repository files verified successfully!')\n",
    "else:\n",
    "    print('⚠️ Warning: data/generate_synthetic_v2.py not found in current working directory!')\n"
]

cell_4_code = [
    "# Step 3: Generate SymPy-validated disjoint synthetic dataset\n",
    "import os\n",
    "if os.path.exists('KLH-csit-2026-2420090008-mathSlm'):\n",
    "    os.chdir('KLH-csit-2026-2420090008-mathSlm')\n",
    "!PYTHONUNBUFFERED=1 python3 data/generate_synthetic_v2.py\n"
]

cell_5_code = [
    "# Step 4: Run Tiny Sanity Test\n",
    "# Verifies pipeline and model capacity to fit small dataset\n",
    "import os\n",
    "if os.path.exists('KLH-csit-2026-2420090008-mathSlm'):\n",
    "    os.chdir('KLH-csit-2026-2420090008-mathSlm')\n",
    "!PYTHONUNBUFFERED=1 python3 scripts/run_tiny_sanity.py\n"
]

cell_6_code = [
    "# Step 5: Execute 4-Way Controlled Diagnostic Matrix (Exp-A, Exp-B, Exp-C, Exp-D)\n",
    "# Runs Exp-A (7.34M Direct 5ep), Exp-B (30M Direct 5ep), Exp-C (7.34M Reasoning 5ep), Exp-D (7.34M Reasoning 1ep)\n",
    "import os\n",
    "if os.path.exists('KLH-csit-2026-2420090008-mathSlm'):\n",
    "    os.chdir('KLH-csit-2026-2420090008-mathSlm')\n",
    "!PYTHONUNBUFFERED=1 python3 scripts/run_controlled_matrix.py auto\n"
]

nb["cells"][1]["source"] = cell_2_code
nb["cells"][3]["source"] = cell_4_code
nb["cells"][4]["source"] = cell_5_code
nb["cells"][5]["source"] = cell_6_code

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Updated {nb_path} with robust working directory handling.")
