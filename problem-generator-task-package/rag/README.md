# RAG untuk Pembuat Soal (Problem Generator)

## Tanggung Jawab
Memberikan konteks (Bank Soal, Silabus) kepada LLM/VLLM Pembuat Soal.

## Daftar Tugas (To-Do List)
- [ ] Bangun sistem RAG dengan Bank Soal pseudocode (standar Tel-U).
- [ ] Sertakan Silabus dalam cakupan retrieval.
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **plan**: Rencana belajar.
- **todays date**: Tanggal hari ini.

### Output
```json
{
    "context": "isi_context_from_bank_soal_and_syllabus"
}
```
