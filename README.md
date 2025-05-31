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

Semua kolom memiliki nilai non-null, artinya tidak ada missing values yang perlu ditangani. Dataset ini siap untuk dianalisis lebih lanjut.

Selanjutnya, kita akan menampilkan statistik deskriptif dari dataset menggunakan df.describe(). Fungsi ini memberikan ringkasan statistik untuk kolom-kolom numerik, seperti jumlah data, nilai rata-rata, standar deviasi, serta nilai minimum dan maksimum. Ini membantu kita memahami distribusi data dan mengidentifikasi outlier atau anomali yang mungkin ada.

![3](https://github.com/user-attachments/assets/51950cd5-b0c1-4542-8581-c977b5bf91c3)

Berdasarkan hasil statistik deskriptif yang ditampilkan oleh df.describe(), kita dapat melihat beberapa informasi penting mengenai kolom-kolom numerik dalam dataset.

1. CustomerID
   - Terdiri dari 200 data unik, ID pelanggan bervariasi dari 1 hingga 200.
2. Age (Usia)
   - Usia pelanggan berkisar antara 18 hingga 70 tahun dengan rata-rata 38.85 tahun.
   - Sebagian besar pelanggan berada pada rentang usia 28.75 hingga 49 tahun (kuartil ke-1 hingga ke-3).
3. Annual Income (k$) (Pendapatan Tahunan)
   - Pendapatan tahunan pelanggan bervariasi antara 15 hingga 137 ribu dolar dengan rata-rata 60.56 ribu dolar.
   - Sebagian besar pelanggan memiliki pendapatan tahunan antara 41.5 hingga 78 ribu dolar.
4. Spending Score (1–100) (Skor Pengeluaran)
   - Skor pengeluaran pelanggan bervariasi dari 1 hingga 99 dengan rata-rata skor pengeluaran sebesar 50.2.
   - Sebagian besar pelanggan memiliki skor pengeluaran antara 34.75 hingga 73.

## Exploratory Data Analysis

Tahap ketiga yang paling penting dalam analisis data adalah exploratory data analysis (EDA). Pada tahap ini, kita melakukan eksplorasi mendalam terhadap dataset untuk memahami pola, hubungan, dan anomali yang ada. EDA memungkinkan kita untuk mendapatkan wawasan awal yang penting untuk analisis lebih lanjut dan mempersiapkan data sebelum membangun model. 

Aktivitas utama dalam EDA mencakup visualisasi data melalui grafik dan plot untuk melihat distribusi serta hubungan antar fitur, analisis korelasi dalam mengidentifikasi hubungan antara fitur-fitur numerik, serta deteksi anomali dan outlier yang dapat memengaruhi model.

Untuk memvisualisasikan distribusi gender pada dataset, kita menghitung jumlah masing-masing kategori gender menggunakan value_counts() dan menampilkan hasilnya dalam diagram lingkaran (pie chart). Diagram ini, yang dihasilkan dengan plt.pie(), menggambarkan proporsi gender dengan label 'Female' dan 'Male', serta menampilkan persentase setiap kategori. Grafik ini memudahkan kita untuk melihat distribusi gender secara visual serta memahami perbandingan antara jumlah wanita dan pria dalam dataset. Hasilnya berikut.

![4](https://github.com/user-attachments/assets/c9ddd6cd-a621-4c9e-aec9-f1142f32fffc)
