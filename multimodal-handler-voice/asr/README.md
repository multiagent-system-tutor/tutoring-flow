# Paket Tugas ASR

## Tanggung Jawab
- Mengimplementasikan Pengenalan Suara Otomatis (ASR) untuk **Bahasa Indonesia**.

## Daftar Tugas (To-Do List)
- [ ] Eksperimen dengan berbagai metode ASR (utamakan sumber daya gratis).
- [ ] Identifikasi tantangan dengan berbagai metode.
- [ ] Ukur kecepatan inferensi.

## Input/Output
- **Input**: Suara/Audio (tipe apapun)
- **Output**:
  ```python
  {
      'time': float,
      'text': str, # Transkripsi
      'voice': ... # Opsional: representasi sinyal/numerik
  }
  ```

## Sumber Daya
- https://huggingface.co/docs/transformers/tasks/asr
- https://github.com/weimeng23/speech-recognition-learning-resources
