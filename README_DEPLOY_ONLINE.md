# Deploy Online — Credit Scoring Trustworthy AI

Paket ini disiapkan agar tugas **Implementasi Credit Scoring Berbasis Trustworthy AI** bisa diakses online oleh dosen.

## Isi Paket

| File | Fungsi |
|---|---|
| `Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.ipynb` | Notebook utama yang sudah presentasi-friendly dan siap dibuka di Google Colab / GitHub |
| `Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.html` | Versi HTML statis untuk dosen yang hanya ingin membaca tanpa menjalankan kode |
| `app.py` | Web app Streamlit interaktif untuk demo dataset, pipeline, model, fairness, dan simulasi prediksi |
| `requirements.txt` | Daftar library Python untuk deploy Streamlit |
| `runtime.txt` | Versi Python yang disarankan untuk Streamlit Cloud |
| `.gitignore` | File pendukung agar repository lebih rapi |

---

## Opsi 1 — Paling Mudah: Upload ke Google Drive

Gunakan opsi ini kalau dosen hanya perlu melihat hasil notebook.

1. Upload file `Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.html` ke Google Drive.
2. Klik kanan file → **Share**.
3. Ubah akses menjadi **Anyone with the link**.
4. Kirim link ke dosen.

Kelebihan: paling cepat dan tampilannya rapi.  
Kekurangan: dosen tidak menjalankan ulang kode secara langsung.

---

## Opsi 2 — Notebook Online via GitHub + Google Colab

Gunakan opsi ini kalau dosen ingin membuka dan menjalankan ulang notebook.

1. Buat repository GitHub baru, misalnya:
   `credit-scoring-trustworthy-ai`
2. Upload file berikut ke repository:
   - `Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.ipynb`
   - `requirements.txt`
3. Setelah upload, buka notebook dari Google Colab dengan format link:

```text
https://colab.research.google.com/github/USERNAME/NAMA_REPO/blob/main/Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.ipynb
```

Contoh jika username GitHub kamu `mahasiswa-ai` dan repo kamu `credit-scoring-trustworthy-ai`:

```text
https://colab.research.google.com/github/mahasiswa-ai/credit-scoring-trustworthy-ai/blob/main/Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.ipynb
```

Kelebihan: dosen bisa melihat alur notebook dan menjalankan ulang cell.  
Kekurangan: perlu upload ke GitHub terlebih dahulu.

---

## Opsi 3 — Web App Interaktif via Streamlit Community Cloud

Gunakan opsi ini kalau dosen ingin mencoba aplikasi web sederhana.

1. Buat repository GitHub baru, misalnya:
   `credit-scoring-trustworthy-ai`
2. Upload seluruh isi paket ini ke repository.
3. Buka Streamlit Community Cloud.
4. Klik **Create app** atau **New app**.
5. Hubungkan ke repository GitHub yang sudah dibuat.
6. Pada bagian main file, isi:

```text
app.py
```

7. Klik deploy.
8. Setelah proses selesai, Streamlit akan memberikan link aplikasi. Link inilah yang bisa dikirim ke dosen.

Kelebihan: dosen bisa mencoba dashboard interaktif.  
Kekurangan: butuh akun Streamlit dan GitHub.

---

## Link yang Bisa Dikirim ke Dosen

Setelah upload/deploy, isi bagian ini secara manual:

```text
Link Notebook Colab:
https://colab.research.google.com/github/USERNAME/NAMA_REPO/blob/main/Implementasi_Credit_Scoring_Trustworthy_AI_online_ready.ipynb

Link Web App Streamlit:
https://NAMA-APP.streamlit.app/

Link HTML/Drive:
ISI_LINK_GOOGLE_DRIVE_DI_SINI
```

---

## Catatan Presentasi

Kalimat yang bisa disampaikan:

> “Kami juga menyiapkan versi online agar dosen dapat mengakses notebook dan web app tanpa perlu membuka file lokal. Notebook berisi alur implementasi lengkap, sedangkan web app menampilkan ringkasan dataset, pipeline preprocessing, hasil training model, evaluasi fairness, dan simulasi prediksi risiko kredit.”

---

## Troubleshooting Singkat

### Jika notebook di Colab error karena `fairlearn` belum tersedia
Jalankan cell instalasi berikut di bagian awal notebook:

```python
!pip -q install fairlearn
```

### Jika Streamlit gagal deploy karena dependency
Pastikan file `requirements.txt` berada di folder utama repository, sejajar dengan `app.py`.

### Jika dataset gagal dimuat
Pastikan deployment memiliki akses internet karena dataset diambil dari URL publik:

```text
https://raw.githubusercontent.com/datasets/openml-datasets/main/data/credit-g/credit-g.csv
```


## Catatan Penting Jika Deploy Error di Streamlit

Jika log menunjukkan `Using Python 3.14` lalu gagal pada `scipy`, jangan hanya mengandalkan `runtime.txt`. Saat deploy/redeploy di Streamlit Community Cloud, buka **Advanced settings** lalu pilih **Python 3.11**. File `requirements.txt` di paket ini sudah dipin agar dependency machine learning lebih stabil.

Lihat juga file `README_FIX_STREAMLIT_ERROR.md` untuk langkah perbaikan rinci.
