# Latihan Studi Kasus: Penerapan Algoritma Clustering untuk Mengelompokkan Data Customer Mall

### Platform : Dicoding

### Kelas : Belajar Machine Learning untuk Pemula

### Modul : Unsupervised Learning - Clustering

### Dataset : [Kaggle-Mall Customer Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Data Loading

Data Mall Customer ini biasanya digunakan untuk mengeksplorasi perilaku belanja pelanggan berdasarkan berbagai fitur, seperti usia, jenis kelamin, pendapatan tahunan, dan skor pengeluaran. Dengan menggunakan data ini, kita dapat mencoba berbagai teknik clustering untuk mengelompokkan pelanggan ke dalam segmen-segmen yang berbeda berdasarkan karakteristik mereka.

![1](https://github.com/user-attachments/assets/e81f0e6f-ce7c-4711-a29d-65c239f7ed1f)

Selanjutnya, akan ditampilkan informasi umum tentang dataset menggunakan df.info(). Ini akan memberikan gambaran mengenai jumlah baris dan kolom, tipe data setiap kolom, serta jumlah nilai non-null yang ada. Informasi ini penting untuk memahami struktur dataset dan memastikan tidak ada missing values yang perlu ditangani.

![2](https://github.com/user-attachments/assets/50fa61b9-0174-441c-8521-eb2a849693a5)

Dari hasil output df.info(), kita dapat melihat bahwa dataset ini terdiri atas 200 baris dan 5 kolom. Berikut adalah detail dari setiap kolom.

1. CustomerID: Ini berisi ID unik untuk setiap pelanggan, bertipe data int64.
2. Gender: Ini menunjukkan jenis kelamin pelanggan, bertipe data object (kategori).
3. Age: Ini menampilkan usia pelanggan dalam tahun, bertipe data int64.
4. Annual Income (k$): Ini berisi pendapatan tahunan pelanggan dalam ribuan dolar, bertipe data int64.
5. Spending Score (1-100): Ini menunjukkan skor pengeluaran pelanggan, mulai dari 1 hingga 100, bertipe data int64.
