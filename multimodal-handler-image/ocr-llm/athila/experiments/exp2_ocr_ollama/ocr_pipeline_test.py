from paddleocr import PaddleOCR
import ollama
import os

# --- Configuration ---
# Image path
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/tesgambar.png"))

print(f"Target Image: {image_path}")
if not os.path.exists(image_path):
    print("Error: Image file not found!")
    exit(1)

# --- Step 1: OCR with PaddleOCR ---
print("\n[Step 1] Running OCR (PaddleOCR)...")
# Initialize PaddleOCR (downloads model automatically on first run)
# lang='en' handles both English and numbers well. 'id' is also available but 'en' is usually sufficient for general latin script.
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 

try:
    result = ocr.ocr(image_path, cls=True)
except Exception as e:
    print(f"OCR Failed: {e}")
    exit(1)

# Extract text only
extracted_lines = []
if result and result[0]:
    for line in result[0]:
        # line format: [[box_coords], (text, confidence)]
        text = line[1][0]
        extracted_lines.append(text)

raw_text = "\n".join(extracted_lines)
print("\n--- OCR Extracted Text ---")
print(raw_text)
print("--------------------------")


# --- Step 2: Processing with LLM (Ollama) ---
print("\n[Step 2] Sending text to LLM (Ollama) for extraction/formatting...")

prompt = f"""
Berikut adalah teks mentah hasil OCR dari sebuah gambar jadwal/daftar tugas.
Tolong rapikan data ini menjadi format JSON yang valid.
Strukturnya harus memiliki key "tasks" yang berisi list object dengan field: "task_name", "description" (jika ada), dan "deadline" (jika ada).

RAW TEXT:
{raw_text}

OUTPUT JSON ONLY:
"""

print("Prompt sent to LLM. Streaming response...\n")

try:
    # Using 'llava' model as a text-llm (it works fine for this) or use 'llama3' if available.
    stream = ollama.chat(
        model='llava', 
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    full_response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response += content
    print("\n\n[Done]")

except Exception as e:
    print(f"\nLLM Error: {e}")
    print("Tip: Ensure 'ollama serve' is running and you have the model installed.")
