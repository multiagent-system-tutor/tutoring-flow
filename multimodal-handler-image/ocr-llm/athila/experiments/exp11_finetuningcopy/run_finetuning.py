import subprocess
import sys
import os

# Fungsi untuk menjalankan Fine-Tuning PaddleOCR via Python
def run_finetuning(config_path, pretrain_path, save_dir='output/rec_finetune'):
    # Cek apakah config ada
    if not os.path.exists(config_path):
        print(f"Error: Config file tidak ditemukan di {config_path}")
        return

    print(f"[INFO] Menggunakan Config: {config_path}")
    print(f"[INFO] Pretrained Model: {pretrain_path}")

    # Susun command argumen
    # Menggunakan sys.executable agar memakai python environment saat ini
    cmd = [
        sys.executable, "-m", "paddleocr.tools.train",
        "-c", config_path,
        "-o",
        f"Global.pretrained_model={pretrain_path}",
        f"Global.save_model_dir={save_dir}",
        "Train.dataset.data_dir=./",
        "Train.dataset.label_file_list=['dataset_lists/rec_gt_train.txt']",
        "Eval.dataset.data_dir=./",
        "Eval.dataset.label_file_list=['dataset_lists/rec_gt_test.txt']"
    ]
    
    print("Memulai Fine-tuning... (Output akan muncul di bawah)")
    print(f"Command Eksekusi: {' '.join(cmd)}\n")
    print("-" * 50)
    
    # Eksekusi process dengan streaming output
    try:
        # bufsize=1 means line buffered
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Baca output baris per baris
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
             print("\n" + "="*50)
             print("[SUCCESS] Fine-tuning Berhasil! Model tersimpan di", save_dir)
             print("="*50)
        else:
             print(f"\n[FAILURE] Training gagal dengan exit code {process.returncode}")
    except Exception as e:
        print(f"Gagal menjalankan training: {e}")

if __name__ == "__main__":
    # ==========================================
    # KONFIGURASI PATH (WAJIB DIISI)
    # ==========================================
    # Masukkan Absolute Path ke file .yml config kamu
    # Contoh: r"F:\projek dosen\tutoring\...\Configs\en_PP-OCRv4_rec.yml"
    CONFIG_PATH = r"MASUKKAN_PATH_CONFIG_KAMU_DISINI.yml"
    
    # Masukkan Absolute Path ke folder pretrain model kamu
    # Contoh: r"F:\projek dosen\tutoring\...\Pretrain\en_PP-OCRv4_rec_train\best_accuracy"
    PRETRAIN_PATH = r"MASUKKAN_PATH_PRETRAIN_MODEL_KAMU_DISINI"

    # Jalankan
    if "MASUKKAN" in CONFIG_PATH:
        print("Tolong edit file ini (run_finetuning.py) dan isi CONFIG_PATH serta PRETRAIN_PATH dengan benar dahulu.")
    else:
        run_finetuning(CONFIG_PATH, PRETRAIN_PATH)
