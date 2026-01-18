import ollama
import os

# Pastikan path image benar. 
# Script ini ada di experiments/exp2_ocr_ollama/
# Dataset ada di dataset/
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/tesgambar.png"))

print(f"Target Image: {image_path}")

if not os.path.exists(image_path):
    print("Error: Image file not found!")
    exit(1)

try:
    print("Sending request to Ollama (model: llava)...")
    print("Note: Pastikan sudah run 'ollama pull llava' di terminal.")
    print("Processing... (This might take a minute on the first run)\n")

    stream = ollama.chat(
        model='llava',
        messages=[
            {
                'role': 'user',
                'content': 'Read the text in this image. Output the text exactly as it appears.',
                'images': [image_path]
            }
        ],
        stream=True
    )
    
    print("--- Result ---")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")

except Exception as e:
    print(f"\nError occurred: {e}")
    if "not found" in str(e) or "pull" in str(e):
        print("Tip: Sepertinya model 'llava' belum ada. Coba jalankan: ollama pull llava")
