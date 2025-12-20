# Pembuat Soal (Problem Generator LLM)

## Tanggung Jawab
Menghasilkan soal pseudocode berbasis teks yang terkait dengan konteks RAG.

## Daftar Tugas (To-Do List)
- [ ] Desain LLM untuk membuat soal.
- [ ] Pastikan gaya pseudocode sesuai dengan standar Tel-U.
- [ ] Validasi apakah LLM dapat menghasilkan soal + solusi berdasarkan konteks RAG.
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **plan**: Rencana belajar.
- **todays date**: Tanggal hari ini.

### Output
```json
{
    "soal": "text_based_problem",
    "solution": "solution_key"
}
```
