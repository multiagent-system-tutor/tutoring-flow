# Agen Perencana (Planner Agent)

## Tanggung Jawab
Membuat rencana belajar (study plan) untuk siswa.
- **Agen**: Teacher Agent, Reflective Agent (dengan agen tambahan jika diperlukan).
- **Prompting**: Self-critic menggunakan teknik CRITIC.

## Daftar Tugas (To-Do List)
- [ ] Desain Planner Agent.
- [ ] Ubah Silabus menjadi format dictionary agar mudah di-parsing.
- [ ] Implementasikan Teacher Agent (CoT/Zero-shot) & Reflective Agent (CRITIC).
- [ ] Opsional: Integrasi YouTube API.
- [ ] Catat waktu inferensi.

## Input/Output

### Input
- **student personal info**: Tingkat pengetahuan, riwayat, dll.
- **syllabus**: Struktur pembelajaran.
- **todays date**: Tanggal hari ini.

### Output
```json
{
    "plan": "generated_study_plan"
}
```

## Sumber Daya
- https://dl.acm.org/doi/pdf/10.1145/3698205.3729541
