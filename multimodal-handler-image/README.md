# Paket Tugas OCR

## Tanggung Jawab
1. **OCR**: Ekstrak teks dari gambar bersih.
2. **OCR + LLM**: Ekstrak teks dan proses/koreksi dengan LLM.
3. **MLLM**: Pemrosesan visi-bahasa end-to-end.

## Daftar Tugas (To-Do List)
- [ ] Uji Ekstraktor Teks pada **Gambar Bersih**.
- [ ] Uji Ekstraktor Teks pada **Gambar Kotor** (Tip-X, coretan).
  - Tujuan: Memahami pola pikir siswa dari coretan.
- [ ] Uji Ekstraktor Teks pada **Gambar Kotor dengan Koreksi** (Panah, Tip-X).
  - Tujuan: Bisakah mesin mengenali koreksi/tanda panah?

## Input/Output
- **Input**: Gambar (ukuran berapapun)
- **Output**:
  ```python
  {
      'time': float,
      'text': str, # Pseudocode yang diekstrak
      'image': ... # Opsional: representasi gambar yang diproses
  }
  ```

## Catatan
- Dokumentasikan teknik prompting jika menggunakan LLM/MLLM.
- **Catat waktu inferensi**.
