# RAG untuk Automated Feedback System (AFS)

## Tanggung Jawab
Memberikan konteks kepada Teacher Agent.
- **Sumber**: Bank Soal, interaksi sebelumnya.

## Daftar Tugas (To-Do List)
- [ ] Desain sistem RAG untuk AFS.
- [ ] Eksplorasi model embedding gratis.
- [ ] Eksplorasi Vector DB (FAISS, Chroma, Pinecone, atau Numpy biasa).
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **student input**: Input/pertanyaan saat ini.
- **summary**: Ringkasan sesi.
- **student profile**: Data siswa.

### Output
```json
{
    "context": "isi_context"
}
```
*Catatan: Konteks output dalam format teks.*

## Sumber Daya
- https://artificialanalysis.ai/
- https://huggingface.co/spaces/mteb/leaderboard
- https://docs.langchain.com/langsmith/evaluation-approaches#retrieval-augmented-generation-rag
