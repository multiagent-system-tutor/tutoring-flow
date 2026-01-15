import json
import os

nb_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp2_ocr_ollama\explore.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Corrected Processing Loop to extracting only message content AND include ground_truth
corrected_processing_code = [
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
    "    ocr_result = ocr.predict(image_path)\n",
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
    "            # FIX: Properly extract content from response object\n",
    "            if isinstance(response, dict) and 'message' in response:\n",
    "                final_text_output = response['message']['content']\n",
    "            elif hasattr(response, 'message') and hasattr(response.message, 'content'):\n",
    "                final_text_output = response.message.content\n",
    "            else:\n",
    "                # Fallback clean up if it's a raw string representation\n",
    "                import re\n",
    "                str_resp = str(response)\n",
    "                match = re.search(r\"content='(.*?)'\", str_resp, re.DOTALL)\n",
    "                if match:\n",
    "                    final_text_output = match.group(1).replace(\"\\\\n\", \"\\n\")\n",
    "                else:\n",
    "                    final_text_output = str_resp # Last resort\n",
    "            \n",
    "            # Clean up markdown code blocks if present\n",
    "            final_text_output = final_text_output.replace(\"```plaintext\", \"\").replace(\"```\", \"\").strip()\n",
    "            \n",
    "        except Exception as e:\n",
    "             print(f\"LLM Error: {e}\")\n",
    "             final_text_output = raw_text # Fallback to raw text if LLM fails\n",
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
    "        'final_text': final_text_output,\n",
    "        'ground_truth': ground_truth_text\n",
    "    })"
]

found_count = 0
for cell in cells:
    source = cell.get('source', [])
    source_text = "".join(source)
    
    if "ocr_result = ocr.predict(image_path)" in source_text:
        cell['source'] = corrected_processing_code
        found_count += 1
        print("Updated Processing Loop with ground_truth field")
        break

print(f"Total updated cells: {found_count}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
