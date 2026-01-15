# Pembuat Soal (Problem Generator VLLM)

## Tanggung Jawab
Menghasilkan soal berbasis gambar atau soal teks dengan gambar pendukung.

## Daftar Tugas (To-Do List)
- [x] Tugas VLLM khusus untuk pembuatan soal.
- [x] Eksplorasi pembuatan soal yang menyertakan gambar untuk membantu pemahaman.
- [x] Catat waktu inferensi.

## Input/Output

### Input
- **plan**: Rencana belajar.
- **todays date**: Tanggal hari ini.

### Output
```json
{
    "soal": "image_or_text_plus_image",
    "solution": "solution_key"
}
```
