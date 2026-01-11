import pandas as pd
import os

# Setup Path
BASE_DIR = r"f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\experiments\exp1_ocr_llm_test"
CSV_PATH = os.path.join(BASE_DIR, "ocr_evaluation_results.csv")
PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, "prompt_correction.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_prompts")

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Read Data
if not os.path.exists(CSV_PATH):
    print(f"Error: File CSV tidak ditemukan di {CSV_PATH}")
    print("Pastikan kamu sudah menjalankan 'explore.ipynb' untuk menghasilkan CSV tersebut.")
    exit()

df = pd.read_csv(CSV_PATH)

# 2. Read Prompt Template
with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template = f.read()

print(f"Generating prompts for {len(df)} entries...\n")

# 3. Generate Prompt Files
for index, row in df.iterrows():
    filename = row['filename']
    ocr_text = row['ocr_raw']
    
    # Fill template
    # Gunakan .format() atau replace manual jika kurung kurawal di template kompleks 
    # (karena template kamu pakai {OCR_TEXT}, replace aman)
    filled_prompt = template.replace("{OCR_TEXT}", str(ocr_text))
    
    # Create TXT file name based on image name
    txt_filename = f"prompt_for_{filename}.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    # Write to file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(filled_prompt)
        
    print(f"[{index+1}/{len(df)}] Generated: {txt_filename}")

print(f"\nSelesai! Cek folder: {OUTPUT_DIR}")
