# Fix Error Deploy Streamlit — scipy / gfortran / Python 3.14

## Masalah yang terjadi

Log deploy menunjukkan Streamlit Cloud memakai Python 3.14.5, lalu mencoba meng-install `scikit-learn` dan dependency `scipy`. Pada proses tersebut, `scipy` terunduh sebagai source package `.tar.gz` dan gagal build karena compiler Fortran seperti `gfortran` tidak tersedia.

Intinya, ini bukan masalah utama di `app.py`, tetapi masalah environment dan dependency saat install package.

## File yang perlu diganti di GitHub

Upload/ganti file berikut di root repository GitHub kamu:

1. `requirements.txt`
2. `runtime.txt`
3. `app.py` jika ingin sekalian menyamakan dengan paket ini

Isi `requirements.txt` sudah dipin agar dependency lebih stabil:

```txt
streamlit==1.39.0
pandas==2.2.3
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.4
fairlearn==0.10.0
```

## Langkah wajib di Streamlit Community Cloud

Karena versi Python tidak cukup diubah dari file repo saja, lakukan ini:

1. Buka dashboard Streamlit Community Cloud.
2. Masuk ke app kamu.
3. Kalau app sudah telanjur memakai Python 3.14, hapus app tersebut dari Streamlit Cloud.
4. Deploy ulang dari repository GitHub yang sama.
5. Saat deploy ulang, klik **Advanced settings**.
6. Pilih **Python 3.11**.
7. Pastikan main file path tetap `app.py`.
8. Klik deploy.

## Kenapa harus Python 3.11?

Package machine learning seperti `scikit-learn` dan `scipy` lebih aman dijalankan pada Python 3.11 untuk setup tugas ini. Dengan Python 3.11 dan versi package yang dipin, Streamlit Cloud seharusnya mengambil prebuilt wheel, bukan membangun `scipy` dari source.

## Setelah push update

Di GitHub:

1. Buka repository kamu.
2. Upload file dari paket fix ini.
3. Commit changes.
4. Kembali ke Streamlit Cloud.
5. Redeploy / reboot app.

Jika masih error, screenshot/log terbaru perlu dicek lagi karena kemungkinan error berikutnya sudah berbeda dari error dependency awal ini.
