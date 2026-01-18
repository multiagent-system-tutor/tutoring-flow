import json
import os

nb_path = r"f:/projek dosen/tutoring/tutoring-flow/multimodal-handler-image/ocr-llm/athila/experiments/exp3_ocr_ollama_dataset_soal2/explore.ipynb"

def modify_notebook_for_visualization():
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} not found")
        return

    print(f"Reading {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Add visualization cell code
    vis_code = [
        "# --- STAGE 1: OCR ---\n",
        "# Use predict() as ocr() is deprecated and fails with cls arg\n",
        "ocr_result = ocr.predict(image_path)\n",
        "extracted_lines = []\n",
        "bboxes = []\n",
        "\n",
        "if ocr_result and len(ocr_result) > 0:\n",
        "    if isinstance(ocr_result[0], dict) and \"rec_texts\" in ocr_result[0]:\n",
        "        extracted_lines = ocr_result[0].get(\"rec_texts\", [])\n",
        "        # Handle bbox extraction if available in this format (dt_polys)\n",
        "        if \"dt_polys\" in ocr_result[0]:\n",
        "            bboxes = ocr_result[0][\"dt_polys\"]\n",
        "            \n",
        "    elif isinstance(ocr_result[0], list):\n",
        "        # Standard [[[[x,y],..], (text, conf)], ...]\n",
        "        # Or simplified\n",
        "        for line in ocr_result[0]:\n",
        "            if isinstance(line, list) and len(line) >= 2:\n",
        "                # BBox is usually index 0, text info index 1\n",
        "                if isinstance(line[0], list): # Polygon points\n",
        "                    bboxes.append(line[0])\n",
        "                \n",
        "                if isinstance(line[1], tuple) or isinstance(line[1], list):\n",
        "                    extracted_lines.append(line[1][0])\n",
        "\n",
        "raw_text = \"\\n\".join(extracted_lines)\n",
        "\n",
        "# --- Visualization & Saving ---\n",
        "if bboxes:\n",
        "    # Visualize BBoxes\n",
        "    import cv2\n",
        "    import matplotlib.pyplot as plt\n",
        "    \n",
        "    img_vis = cv2.imread(image_path)\n",
        "    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)\n",
        "    \n",
        "    for box in bboxes:\n",
        "        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))\n",
        "        cv2.polylines(img_vis, [box], True, (255, 0, 0), 2)\n",
        "    \n",
        "    # Save visualization\n",
        "    vis_filename = f\"vis_{filename}\"\n",
        "    vis_path = os.path.join(r'results/bbox', vis_filename)\n",
        "    # Convert back to BGR for opencv saving if using cv2.imwrite, or just use plt\n",
        "    # Let's use plt to save as is simpler with established RGB\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.imshow(img_vis)\n",
        "    plt.axis('off')\n",
        "    plt.savefig(vis_path, bbox_inches='tight')\n",
        "    plt.close()\n",
        "    print(f\"  Saved bbox visualization to {vis_path}\")\n",
        "    \n",
        "    # Save bbox coordinates txt\n",
        "    bbox_txt_path = os.path.join(r'results/bbox', f\"bbox_{filename_base}.txt\")\n",
        "    with open(bbox_txt_path, 'w', encoding='utf-8') as f:\n",
        "        for i, box in enumerate(bboxes):\n",
        "            text = extracted_lines[i] if i < len(extracted_lines) else \"\"\n",
        "            f.write(f\"{box} | {text}\\n\")\n"
    ]

    # Find the main processing loop and replace the OCR section
    found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            start_idx = -1
            end_idx = -1
            
            # Look for markers
            for idx, line in enumerate(source):
                if "# --- STAGE 1: OCR ---" in line:
                    start_idx = idx
                if "# --- STAGE 2: LLM ---" in line:
                    end_idx = idx
            
            if start_idx != -1 and end_idx != -1:
                print(f"Found OCR processing block in cell {i}.")
                
                # Keep everything before Stage 1
                new_source = source[:start_idx]
                # Insert new visualization code
                new_source.extend(vis_code)
                # Keep everything from Stage 2 onwards
                new_source.extend(source[end_idx:])
                
                cell['source'] = new_source
                found = True
                break
    
    # 2. Modify results export path to 'results/'
    if found:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = cell['source']
                new_source = []
                changed = False
                for line in source:
                    if "to_csv('exp2_results.csv'" in line:
                        new_source.append(line.replace("'exp2_results.csv'", "'results/exp3_results.csv'"))
                        changed = True
                    elif "savefig(save_path" in line and "cer_comparison.png" in ''.join(source):
                         # Usually save_path is defined earlier
                         pass
                         new_source.append(line)
                    elif "save_path = 'cer_comparison.png'" in line:
                         new_source.append("        save_path = 'results/cer_comparison.png'\n")
                         changed = True
                    else:
                        new_source.append(line)
                
                if changed:
                    cell['source'] = new_source
                    print(f"Updated paths in cell {i}")

        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated with visualization and path changes.")
    else:
        print("Error: Could not find the OCR processing block to inject visualization code.")

if __name__ == "__main__":
    modify_notebook_for_visualization()
