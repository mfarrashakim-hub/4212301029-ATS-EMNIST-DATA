# EMNIST Letters Classification using HOG + SVM + LOOCV

Project ini merupakan implementasi sistem klasifikasi huruf tulisan tangan menggunakan dataset **EMNIST Letters**, dengan metode **HOG Feature Extraction**, algoritma **Support Vector Machine (SVM)**, dan evaluasi menggunakan **Leave-One-Out Cross Validation (LOOCV)** pada total **13.000 sampel**.

Repository ini berisi source code utama dalam bentuk **Jupyter Notebook (.ipynb)** yang digunakan untuk:
- Preprocessing dan perbaikan orientasi citra EMNIST
- Ekstraksi fitur menggunakan HOG
- Tuning parameter SVM dengan GridSearchCV
- Pelatihan model SVM
- Evaluasi performa menggunakan LOOCV
- Visualisasi confusion matrix dan metrik evaluasi lainnya

---

##  **Struktur Repository**


> Catatan:  
> File besar seperti `X_hog.npy`, atau `loocv_13k_ckpt.pkl` **tidak di-upload** karena termasuk file hasil proses, bukan source code.

---

##  **1. Dataset**

Dataset yang digunakan:
- **EMNIST Letters (Balanced)**  
- Sumber: https://www.kaggle.com/datasets/crawford/emnist/data  
- 26 kelas (huruf A–Z)
- Tiap gambar berukuran **28×28 piksel** dalam format flattened (784 kolom)

Untuk tugas ini, digunakan:
500 sampel per kelas → total **13.000 data**  
Data diambil secara random & seimbang

---

## **2. Preprocessing**

Tahapan preprocessing meliputi:

- Membaca dataset dalam format CSV  
- Membentuk ulang gambar menjadi 28×28  
- **Perbaikan orientasi EMNIST** dengan:
  - flip vertical  
  - transpose  
  - flip horizontal  
- Normalisasi piksel  
- Pengacakan (shuffle) dataset

Perbaikan orientasi sangat penting untuk mendapatkan fitur HOG yang akurat.

---

##  **3. Ekstraksi Fitur – HOG (Histogram of Oriented Gradients)**

Fitur HOG digunakan untuk mengekstrak pola tepi dan struktur huruf.

Parameter HOG:
- `orientations = 9`
- `pixels_per_cell = (8, 8)`
- `cells_per_block = (2, 2)`
- `block_norm = 'L2-Hys'`

Setiap gambar menghasilkan vektor fitur berdimensi ±144 elemen.

---

## **4. Tuning Parameter SVM**

Tuning dilakukan menggunakan **GridSearchCV** pada subset data sebanyak **1.000 sampel**, dengan parameter:

- Kernel: `linear`, `rbf`
- C: `1`, `5`, `10`
- Gamma: `scale`, `0.01`, `0.001`

Parameter terbaik kemudian disimpan dan digunakan untuk proses LOOCV pada 13.000 data.

---

## **5. Evaluasi Menggunakan LOOCV (Leave-One-Out Cross Validation)**

LOOCV dipilih karena memberikan evaluasi yang sangat ketat.

Pada dataset 13.000 sampel:
- Total iterasi = **13.000 training SVM**
- Setiap iterasi:
  - 1 sampel sebagai testing
  - 12.999 sampel sebagai training

Untuk mencegah hilangnya progress:
Digunakan sistem **checkpoint otomatis**  
Proses dapat dilanjutkan jika notebook dimatikan  
Hasil akhir disimpan sebagai `.npy`

---

##  **6. Hasil Evaluasi**

Metrik evaluasi yang digunakan:
- **Accuracy**
- **Precision (macro-average)**
- **Recall (macro-average)**
- **F1-score (macro-average)**
- **Confusion Matrix (normalized)**

Contoh output:
- Akurasi total: **~86–87%**
- Performa tiap kelas (huruf A–Z) ditampilkan dalam classification report

---

## **7. Visualisasi**

Notebook menyertakan:
- Heatmap Confusion Matrix
- Grafik metrik evaluasi
- Contoh hasil prediksi

---

##  **8. Cara Menjalankan Notebook**

1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn scikit-image matplotlib seaborn tqdm
2.  Jalankan notebook:
    Load dataset
    Lakukan ekstraksi HOG
    Jalankan tuning SVM
    Jalankan LOOCV (proses lama ± 10–20 jam)
    Lihat hasil evaluasi
