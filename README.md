# Repositori Riset Tutoring Flow

Repositori ini dirancang untuk penelitian dan pengembangan sistem tutoring multimodal. Proyek ini mendukung berbagai tugas termasuk Pengenalan Karakter Optik (OCR), Pengenalan Suara Otomatis (ASR), Sistem Umpan Balik Otomatis (AFS), Penilaian (Scoring), dan Pembuatan Soal (Problem Generation).

## Struktur Direktori

Proyek ini mengikuti struktur "Dual-layer Repo" untuk menyeimbangkan eksperimen individu dan integrasi sistem inti.

### Layer 1: Eksperimen (Task Packages)
Setiap komponen memiliki direktori sendiri untuk eksperimen, penelitian, dan pengembangan. Anggota tim bekerja terutama di folder-folder ini.

- **Multimodal Handlers**:
  - `multimodal-handler-image/`: OCR, OCR+LLM, MLLM.
  - `multimodal-handler-voice/`: ASR.
- **Scoring**:
  - `scoring-task-package/`: Sistem Scoring Multi-Agent.
- **AFS (Automated Feedback System)**:
  - `afs-task-package/`: Multi-agent (Teacher + Reflective), RAG.
- **Planner**:
  - `planner-task-package/`: Agen Perencana Studi (Study Planner Agent).
- **Problem Generator**:
  - `problem-generator-task-package/`: LLM, VLLM, dan RAG untuk pembuatan soal.

### Layer 2: Komponen Inti (`src/components/`)
Kode yang stabil dan siap produksi akan dipromosikan ke layer ini. Komponen-komponen ini menyediakan antarmuka standar untuk integrasi pipeline.

- `src/components/`: Berisi kelas seperti `OCRComponent`, `ASRComponent`, dll.
- `src/utils/`: Utilitas untuk perhitungan waktu (timing), I/O, dll.

## Antarmuka Komponen (Interface)

Semua komponen di `src/components/` harus mematuhi antarmuka berikut:

```python
class ComponentName:
    def __init__(self, config):
        pass

    def run(self, input_data):
        # mengembalikan dict dengan format standar
        pass
```

Format output standar:
```python
{
    'time': float,  # Waktu inferensi
    'text': str,    # Hasil utama
    # field spesifik lainnya
}
```

## Cara Penggunaan (Setup)

1. Instal dependensi (lihat `requirements.txt` di masing-masing paket).
2. Untuk menjalankan pipeline, gunakan skrip di root atau folder tugas tertentu.


## Panduan Pengembangan & Workflow

### 1️⃣ .ipynb DIPAKAI BUAT APA?
**Eksplorasi & Eksperimen Awal**

Dipakai di:
- Mencoba library OCR A vs B
- Tuning prompt LLM
- Cek output sementara
- Bandingin inference time
- EDA data

📌 **Cocok ditaruh di:**
```text
multimodal-handler-image/
└── ocr/
    └── experiments/
        └── exp1_clean_image/
            └── explore.ipynb
```

**KENAPA ipynb?**
- Bisa jalan per cell
- Gampang debug
- Enak buat riset
- Dosen juga ngerti notebook

❌ **TAPI JANGAN:**
- Import notebook ke notebook
- Bikin pipeline dari notebook
- Menaruh logic inti di notebook

### 2️⃣ .py DIPAKAI BUAT APA?
**Implementasi Komponen & Integrasi Pipeline**

Dipakai di `src/components/` (misal: `ocr.py`, `asr.py`, `scoring.py`, `afs_multiagent.py`).

Ini yang:
- Bisa di-import
- Bisa dirangkai jadi pipeline
- Bisa dipanggil dari file lain
- OOP Python

**Contoh pemakaian:**
```python
from src.components.ocr import OCRComponent

ocr = OCRComponent(config)
result = ocr.run(image)
```

📌 *Ini yang dimaksud "biar import antar file kalau pipelinenya jadi enak".*

### 3️⃣ FLOW KERJA YANG BENAR (PENTING)

Urutan idealnya:

**STEP 1 — Eksplor (Notebook)**
Lokasi: `experiments/exp1_clean_image/explore.ipynb`
- Nyoba model
- Nyatet hasil
- Logging inference time
- Simpan hasil ke .json

**STEP 2 — Stabil**
Kalau output sudah konsisten dan metode oke:
➡️ **PORTING KE .py** (contoh: pindahkan logic ke `src/components/ocr.py`)

**STEP 3 — Integrasi**
Pipeline tinggal dirangkai:
`ocr → scoring → afs → planner → problem_generator`

## Referensi

- `keinginankelompok.txt`: Struktur inti dan kesepakatan.
- `pembagiantugas.txt`: Pembagian tugas.
- `isipdf.txt`: Persyaratan teknis rinci dan pernyataan masalah.
