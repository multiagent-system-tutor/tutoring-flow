import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os

# Nama model di Hugging Face
MODEL_NAME = "lightonai/LightOnOCR-2-1B"

def main():
    print(f"Loading model: {MODEL_NAME}...")
    try:
        # Load Processor dan Model
        # Note: Menggunakan device_map="auto" agar otomatis menggunakan GPU jika ada
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Gagal load model. Pastikan koneksi internet lancar dan library transformers terbaru.")
        print(f"Error: {e}")
        return

    # Load Gambar
    image_path = "grafikmodel.png"
    if not os.path.exists(image_path):
        print(f"File gambar {image_path} tidak ditemukan.")
        # Coba cari gambar lain jika ada, atau exit
        return
    
    image = Image.open(image_path).convert("RGB")
    print(f"Memproses gambar: {image_path}")

    # Siapkan input untuk model
    # Prompt standar untuk OCR biasanya cukup sederhana atau kosong tergantung model,
    # tapi LightOnOCR dilatih untuk langsung convert image to text.
    # Kita gunakan format prompt standar Qwen2-VL jika base-nya Qwen, atau instruksi OCR sederhana.
    
    # Berdasarkan info umum LightOnOCR, kita kirim gambar saja dengan prompt instruksi minimal.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe this document into Markdown."} # Prompt umum OCR
            ]
        }
    ]

    # Preprocess inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate Output
    print("Sedang melakukan OCR...")
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=False  # Greedy search biasanya lebih baik untuk OCR agar akurat
    )
    
    # Decode hasil
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    # Ambil bagian respon asisten (karena output bisa berisi prompt juga)
    # Biasanya perlu parsing sedikit, tapi kita print dulu semua.
    print("\n--- Hasil OCR ---\n")
    print(generated_text)
    print("\n-----------------\n")

if __name__ == "__main__":
    main()
