import json
import os

nb_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp2_ocr_ollama\explore.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Update 1: Prompt Description
new_prompt_desc = [
    "**Prompting Technique:**  \n",
    "Kami menggunakan **Structured Instruction Prompting** (mengadopsi elemen Chain-of-Thought / Task Decomposition). Prompt dirancang dengan memberikan konteks yang jelas, definisi masalah, dan langkah-langkah penyelesaian tugas secara berurutan untuk memandu model memperbaiki typo dan struktur tanpa mengubah logika program.\n",
    "\n",
    "**Prompt Specs:**\n",
    "- **Role:** Text Corrector / Coding Assistant.\n",
    "- **Input:** Raw OCR Text.\n",
    "- **Output:** Plaintext Pseudocode (No Markdown, algoritma corrected).\n",
    "- **Strategy:** Step-by-step correction instructions (Fix Typos -> Reformat -> Validate)."
]

# Update 3: Summary
new_summary_code = [
    "print(\"\\n=== SUMMARY ===\")\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "df = pd.DataFrame(results)\n",
    "if not df.empty:\n",
    "    print(f\"Average Time: {df['time'].mean():.4f}s\")\n",
    "    print(f\"Average CER (Raw): {df['cer_raw'].mean():.2%}\")\n",
    "    print(f\"Average CER (Refined): {df['cer_refined'].mean():.2%}\")\n",
    "    print(\"\\nDetailed Results exported to 'exp2_results.csv'\")\n",
    "    df.to_csv('exp2_results.csv', index=False)\n",
    "    \n",
    "    # --- Visualization ---\n",
    "    try:\n",
    "        plt.figure(figsize=(12, 6))\n",
    "        df_melted = df.melt(id_vars=['filename'], value_vars=['cer_raw', 'cer_refined'], var_name='Stage', value_name='CER')\n",
    "        \n",
    "        sns.barplot(data=df_melted, x='filename', y='CER', hue='Stage', palette='viridis')\n",
    "        plt.title('Comparison of CER: Raw OCR vs LLM Refinement')\n",
    "        plt.xlabel('Filename')\n",
    "        plt.ylabel('Character Error Rate (0.0 - 1.0)')\n",
    "        plt.xticks(rotation=45, ha='right')\n",
    "        plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "    except Exception as e:\n",
    "        print(f\"Error plotting graph: {e}\")\n",
    "else:\n",
    "    print(\"No results to show.\")"
]

# Update 2: Processing Loop (Full Cell content)
full_processing_cell_source = [
    "image_files = glob.glob(os.path.join(IMAGES_DIR, \"*.jpg\")) + glob.glob(os.path.join(IMAGES_DIR, \"*.png\")) + glob.glob(os.path.join(IMAGES_DIR, \"*.jpeg\"))\n",
    "print(f\"Found {len(image_files)} images.\")\n",
    "\n",
    "results = []\n",
    "\n",
    "# Ensure OCR is initialized correctly\n",
    "if 'ocr' not in locals():\n",
    "    ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=True)\n",
    "\n",
    "for image_path in image_files:\n",
    "    filename = os.path.basename(image_path)\n",
    "    filename_base = os.path.splitext(filename)[0]\n",
    "    ground_truth_text = read_ground_truth(filename_base)\n",
    "    print(f\"\\nProcessing: {filename}...\")\n",
    "    \n",
    "    start_time = time.time()\n",
    "    \n",
    "    # --- STAGE 1: OCR ---\n",
    "    # Pass path directly to avoid CV2 nuances\n",
    "    ocr_result = ocr.predict(image_path)\n",
    "\n",
    "    extracted_lines = []\n",
    "\n",
    "    if ocr_result and len(ocr_result) > 0:\n",
    "        extracted_lines = ocr_result[0].get(\"rec_texts\", [])\n",
    "\n",
    "    raw_text = \"\\n\".join(extracted_lines)\n",
    "    \n",
    "    # --- STAGE 2: LLM ---\n",
    "    final_text_output = \"\"\n",
    "    if raw_text.strip():\n",
    "        # Load prompt correction\n",
    "        try:\n",
    "            with open('prompt_correction.txt', 'r', encoding='utf-8') as f:\n",
    "                prompt_template = f.read()\n",
    "            prompt_content = prompt_template.replace(\"{OCR_TEXT}\", raw_text)\n",
    "        except Exception as e:\n",
    "            print(f\"Error loading prompt: {e}\")\n",
    "            prompt_content = f\"Correct this OCR text:\\n{raw_text}\"\n",
    "        try:\n",
    "            response = ollama.chat(\n",
    "                model='llava',\n",
    "                messages=[{'role': 'user', 'content': prompt_content}]\n",
    "            )\n",
    "            if isinstance(response, dict) and 'message' in response:\n",
    "                final_text_output = response['message']['content']\n",
    "            else:\n",
    "                final_text_output = str(response)\n",
    "        except Exception as e:\n",
    "             print(f\"LLM Error: {e}\")\n",
    "    \n",
    "    end_time = time.time()\n",
    "    inference_time = end_time - start_time\n",
    "    \n",
    "    # Calculate CER for both stages\n",
    "    cer_raw = calculate_cer(ground_truth_text, raw_text)\n",
    "    cer_refined = calculate_cer(ground_truth_text, final_text_output)\n",
    "    \n",
    "    print(f\"  OCR Length: {len(raw_text)} chars | CER Raw: {cer_raw:.2%} | CER Refined: {cer_refined:.2%} | Time: {inference_time:.2f}s\")\n",
    "    \n",
    "    results.append({\n",
    "        'filename': filename,\n",
    "        'time': inference_time,\n",
    "        'cer_raw': cer_raw,\n",
    "        'cer_refined': cer_refined,\n",
    "        'raw_text': raw_text,\n",
    "        'final_text': final_text_output\n",
    "    })"
]

found_count = 0
for cell in cells:
    source = cell.get('source', [])
    source_text = "".join(source)
    
    if "Zero-Shot Instruction Prompting" in source_text:
        cell['source'] = new_prompt_desc
        found_count += 1
        print("Updated Prompt Description")
    
    elif "cer_score = calculate_cer(ground_truth_text, raw_text)" in source_text:
        cell['source'] = full_processing_cell_source
        found_count += 1
        print("Updated Processing Loop")
        
    elif "print(\"\\n=== SUMMARY ===\")" in source_text:
        cell['source'] = new_summary_code
        found_count += 1
        print("Updated Summary")

print(f"Total updated cells: {found_count}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
