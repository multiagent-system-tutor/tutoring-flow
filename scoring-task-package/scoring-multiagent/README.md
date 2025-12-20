# Sistem Scoring Multi-Agent

## Tanggung Jawab
Mendesain sistem multi-agent untuk menilai pseudocode siswa.
- **Agen**: Style Checker, Logic Checker, Scoring Supervisor.
- **Interaksi**: Iterasi ganjil (1, 3, 5, dst.) antar agen.
- **Prompting**: Zero-shot atau Chain-of-Thought (CoT).

## Daftar Tugas (To-Do List)
- [ ] Desain arsitektur multi-agent (Style Checker, Logic Checker, Supervisor).
- [ ] Eksplorasi berbagai LLM (gratis/bisa di-fine-tune).
- [ ] Implementasikan loop interaksi.
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **pseudocode**: Kode/pseudocode siswa.
- **problem**: Soal yang diberikan.
- **solution**: Solusi referensi.
- **rubric**: Rubrik penilaian (bisa dummy dulu).

### Output
```json
[
    {"score": "value_scorenya"},
    {"correct": "true/false"},
    {"summary": "some_summary_here"},
    {"Misconceptions": "misconceptionsnya apa"}
]
```

## Sumber Daya
- https://arxiv.org/pdf/2503.20851
- https://academy.langchain.com/courses/foundation-introduction-to-langchain-python
