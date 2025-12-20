# Sistem AFS Multi-Agent (Teacher + Reflective)

## Tanggung Jawab
Mendesain sistem multi-agent untuk memberikan umpan balik (feedback) kepada siswa.
- **Agen**: Teacher Agent, Reflective Agent.
- **Interaksi**: Iterasi ganjil.
- **Prompting**: 
  - Reflective Agent: Prompting **CRITIC**.
  - Teacher Agent: Zero-shot, CoT, dll.

## Daftar Tugas (To-Do List)
- [ ] Desain sistem multi-agent (Teacher + Reflective).
- [ ] Implementasikan prompting CRITIC untuk Reflective Agent.
- [ ] Eksplorasi LLM.
- [ ] Integrasikan dengan konteks RAG.
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **student score**: Angka skor.
- **student profile**: Data/riwayat siswa.
- **problem**: Soal.
- **transcription**: Transkripsi lisan & tulisan tangan.
- **misconceptions**: Miskonsepsi yang terdeteksi.
- **RAG context**: Teks dari sistem RAG.

### Output
```json
{
    "Score": "score",
    "summary": "summary"
}
```

## Sumber Daya
- https://academy.langchain.com/courses/foundation-introduction-to-langchain-python
- https://colab.research.google.com/drive/14ND2Fa_Mj5x9d5jOUWdymB_MRcGPtMol#scrollTo=JZfBDhKVtOGu (Kode Indra)
