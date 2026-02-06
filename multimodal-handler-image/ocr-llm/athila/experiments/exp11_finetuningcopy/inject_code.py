import json
import os

nb_path = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore.ipynb'
new_nb_path = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore_updated.ipynb'

# The python code to inject
new_code_source = [
    "# === FINE-TUNING (PYTHON EXECUTION) ===\n",
    "import subprocess\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# ------------------------------------------------------------------\n",
    "# [ACTION REQUIRED] UPDATE THESE PATHS TO YOUR ACTUAL LOCATIONS !!!\n",
    "# ------------------------------------------------------------------\n",
    "# Example: r'F:\\Repo\\PaddleOCR\\configs\\rec\\PP-OCRv4\\en_PP-OCRv4_rec.yml'\n",
    "CONFIG_PATH = r'CHANGE_THIS_TO_YOUR_CONFIG_PATH.yml'\n",
    "\n",
    "# Example: r'F:\\Repo\\PaddleOCR\\pretrain_models\\en_PP-OCRv4_rec_train\\best_accuracy'\n",
    "PRETRAIN_PATH = r'CHANGE_THIS_TO_YOUR_PRETRAIN_MODEL_PATH'\n",
    "\n",
    "def run_finetuning(config_path, pretrain_path, save_dir='output/rec_finetune'):\n",
    "    if 'CHANGE_THIS' in config_path or 'CHANGE_THIS' in pretrain_path:\n",
    "        print(\"STOP: You must edit the CONFIG_PATH and PRETRAIN_PATH variables in this cell first.\")\n",
    "        return\n",
    "\n",
    "    if not os.path.exists(config_path):\n",
    "        print(f\"Error: Config file not found at {config_path}\")\n",
    "        return\n",
    "\n",
    "    print(f\"[INFO] Config: {config_path}\")\n",
    "    print(f\"[INFO] Pretrain: {pretrain_path}\")\n",
    "    \n",
    "    # Construct command\n",
    "    cmd = [\n",
    "        sys.executable, \"-m\", \"paddleocr.tools.train\",\n",
    "        \"-c\", config_path,\n",
    "        \"-o\",\n",
    "        f\"Global.pretrained_model={pretrain_path}\",\n",
    "        f\"Global.save_model_dir={save_dir}\",\n",
    "        \"Train.dataset.data_dir=./\",\n",
    "        \"Train.dataset.label_file_list=['dataset_lists/rec_gt_train.txt']\",\n",
    "        \"Eval.dataset.data_dir=./\",\n",
    "        \"Eval.dataset.label_file_list=['dataset_lists/rec_gt_test.txt']\"\n",
    "    ]\n",
    "    \n",
    "    print(\"-\" * 50)\n",
    "    print(f\"Executing Command:\\n{' '.join(cmd)}\")\n",
    "    print(\"-\" * 50 + \"\\nOutput Stream:\")\n",
    "    \n",
    "    try:\n",
    "        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
    "        for line in iter(process.stdout.readline, ''):\n",
    "            print(line, end='')\n",
    "        process.stdout.close()\n",
    "        process.wait()\n",
    "        if process.returncode == 0:\n",
    "             print(\"\\n[SUCCESS] Fine-tuning Completed.\")\n",
    "        else:\n",
    "             print(f\"\\n[FAILURE] Process exited with code {process.returncode}\")\n",
    "    except Exception as e:\n",
    "        print(f\"Execution failed: {e}\")\n",
    "\n",
    "# Uncomment to run after setting paths\n",
    "# run_finetuning(CONFIG_PATH, PRETRAIN_PATH)\n"
]

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find the cell to replace
    replaced = False
    new_cells = []
    for cell in data['cells']:
        # Check if it's the specific markdown cell
        if cell['cell_type'] == 'markdown':
            source_text = "".join(cell['source'])
            if "FINE-TUNING COMMAND" in source_text or "paddleocr.tools.train" in source_text:
                # Create replacement code cell
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": new_code_source
                }
                new_cells.append(new_cell)
                replaced = True
                print("Found and replaced the Fine-tuning markdown cell.")
                continue
        
        new_cells.append(cell)

    if replaced:
        data['cells'] = new_cells
        # Overwrite the original file
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Successfully updated {nb_path}")
    else:
        print("Target cell not found. No changes made.")

except Exception as e:
    print(f"Error processing notebook: {e}")
