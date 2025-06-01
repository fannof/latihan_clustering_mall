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

Tahap ketiga yang paling penting dalam analisis data adalah exploratory data analysis (EDA). Pada tahap ini, kita melakukan eksplorasi mendalam terhadap dataset untuk memahami pola, hubungan, dan anomali yang ada. EDA memungkinkan kita untuk mendapatkan wawasan awal yang penting untuk analisis lebih lanjut dan mempersiapkan data sebelum membangun model. Aktivitas utama dalam EDA mencakup visualisasi data melalui grafik dan plot untuk melihat distribusi serta hubungan antar fitur, analisis korelasi dalam mengidentifikasi hubungan antara fitur-fitur numerik, serta deteksi anomali dan outlier yang dapat memengaruhi model.

Untuk memvisualisasikan distribusi gender pada dataset, kita menghitung jumlah masing-masing kategori gender menggunakan value_counts() dan menampilkan hasilnya dalam diagram lingkaran (pie chart). Diagram ini, yang dihasilkan dengan plt.pie(), menggambarkan proporsi gender dengan label 'Female' dan 'Male', serta menampilkan persentase setiap kategori. Grafik ini memudahkan kita untuk melihat distribusi gender secara visual serta memahami perbandingan antara jumlah wanita dan pria dalam dataset. Hasilnya berikut.

![4](https://github.com/user-attachments/assets/c9ddd6cd-a621-4c9e-aec9-f1142f32fffc)

Dari pie chart yang ditampilkan, kita dapat ketahui bahwa persentase perempuan lebih besar dibandingkan laki-laki dengan proporsi sebesar 56% untuk perempuan dan 44% untuk laki-laki.

Untuk menganalisis distribusi usia pelanggan, kita mengelompokkan usia ke dalam beberapa kategori dan menghitung jumlah pelanggan pada setiap kategori. Usia dibagi menjadi lima kategori: 18–25, 26–35, 36–45, 46–55, dan 55 ke atas. Setelah menghitung jumlah pelanggan pada setiap kategori, data tersebut digunakan untuk membuat diagram batang (bar chart) yang menunjukkan distribusi usia pelanggan. Proses ini disebut sebagai binning. Ini adalah teknik untuk mengelompokkan nilai-nilai numerik ke dalam interval atau kategori yang disebut bins. Dalam kasus ini, usia pelanggan dikelompokkan ke dalam beberapa rentang usia yang telah ditentukan, dan jumlah pelanggan pada setiap rentang dihitung. Hasilnya kemudian divisualisasikan menggunakan bar chart untuk memudahkan analisis distribusi usia. Binning membantu menyederhanakan data dan memudahkan interpretasi pola-pola dalam dataset. Berikut adalah visualisasi distribusi usia pelanggan berdasarkan kategori yang telah ditentukan.

1. 18–25 tahun: ada 38 pelanggan dalam rentang usia ini.
2. 26–35 tahun: kategori ini memiliki jumlah pelanggan terbanyak, yaitu 60.
3. 36–45 tahun: ada 36 pelanggan dalam rentang usia ini.
4. 46–55 tahun: ada 37 pelanggan yang termasuk dalam kategori ini.
5. 55 tahun ke atas: rentang usia ini memiliki 29 pelanggan.

Visualisasi ini menunjukkan bahwa kelompok usia 26–35 tahun adalah yang terbesar di antara pelanggan, sementara kelompok usia 55 tahun ke atas memiliki jumlah pelanggan paling sedikit. Distribusi usia ini memberikan wawasan penting tentang demografi pelanggan serta dapat membantu dalam perencanaan strategi pemasaran dan layanan.

![5](https://github.com/user-attachments/assets/a2ab350c-a3d4-4e34-81f4-2691b459c292)

Untuk menganalisis distribusi pendapatan tahunan pelanggan, kita mengelompokkan pendapatan ke dalam beberapa kategori dan menghitung jumlah pelanggan pada setiap kategori. Pendapatan tahunan dikelompokkan ke dalam lima rentang.

1. $0–30,000
2. $30,001–60,000
3. $60,001–90,000
4. $90,001–120,000
5. $120,001–150,000

Setelah menghitung jumlah pelanggan dalam setiap kategori, data tersebut divisualisasikan melalui bar chart. Grafik ini memperlihatkan jumlah pelanggan dalam setiap rentang pendapatan dengan warna berbeda untuk masing-masing kategori.

![6](https://github.com/user-attachments/assets/43c61e5b-3d7d-43f3-bbe5-96daeaef0372)

Hasil visualisasi menunjukkan distribusi pendapatan tahunan pelanggan sebagai berikut.

- $0–30,000: ada 32 pelanggan dalam rentang pendapatan ini.
- $30,001–60,000: kategori ini memiliki jumlah pelanggan terbanyak, yaitu 66.
- $60,001–90,000: ada 80 pelanggan dalam rentang pendapatan ini dan menjadikannya kategori dengan jumlah pelanggan terbesar.
- $90,001–120,000: rentang ini memiliki 18 pelanggan.
- $120,001–150,000: kategori ini mencakup 4 pelanggan yang merupakan jumlah paling sedikit di antara semua kategori.

Bar chart ini menunjukkan bahwa pelanggan paling banyak berada dalam rentang pendapatan $60,001–90,000, sedangkan kategori pendapatan tertinggi $120,001–150,000 memiliki jumlah pelanggan yang paling sedikit. Grafik ini memberikan wawasan tentang distribusi pendapatan pelanggan dan dapat membantu dalam merencanakan strategi pemasaran yang lebih efektif.

## Data Splitting

Selanjutnya, kita mengambil dua kolom penting dari dataset: Annual Income (k$) dan Spending Score (1-100). Data dari kedua kolom ini disimpan dalam array X untuk analisis lebih lanjut. Setelah itu, kita menampilkan data yang diambil dalam format DataFrame dengan nama kolom yang sesuai, yaitu Annual Income (k$) dan Spending Score (1-100). Ini memungkinkan kita untuk melihat serta memeriksa nilai-nilai pendapatan tahunan dan skor pengeluaran pelanggan dengan cara yang lebih terstruktur serta mudah dibaca.

Berikut adalah hasil variabel X yang terdiri dari 2 kolom, yaitu Annual Income (k$) dan Spending Score (1-100).

![7](https://github.com/user-attachments/assets/03bbe6e5-c039-4867-8a76-e29046d5fb61)

Dengan data yang telah disiapkan, kita sekarang siap untuk memasuki tahapan pembangunan model clustering. Pada tahap ini, kita akan menggunakan teknik clustering untuk mengelompokkan pelanggan berdasarkan pendapatan tahunan dan skor pengeluaran mereka. 

## Elbow Method

Sebelum melanjutkan ke pembangunan model clustering, kita perlu menentukan jumlah cluster yang optimal untuk data kita. Untuk itu, kita akan menggunakan metode elbow method. Metode ini berfungsi untuk membantu kita memilih jumlah cluster terbaik dengan melihat perubahan total within-cluster sum of squares (WCSS) saat jumlah cluster bertambah. Dengan menggunakan elbow method, kita akan menggambar grafik WCSS terhadap jumlah cluster dan mencari "siku" pada grafik tersebut. Titik letak penurunan WCSS mulai melambat, atau sikunya, biasanya menunjukkan jumlah cluster yang optimal. Ini membantu kita menghindari overfitting dengan memilih jumlah cluster yang sesuai dengan struktur data.

Untuk menentukan jumlah cluster yang optimal, kita menggunakan metode elbow dengan model KMeans. Pertama, kita menginisialisasi model KMeans tanpa menetapkan jumlah cluster awal. Selanjutnya, kita menggunakan KElbowVisualizer untuk mengevaluasi berbagai jumlah cluster dari 1 hingga 10.

![8](https://github.com/user-attachments/assets/2c59c35a-6f67-4432-a7ba-c291acc4c484)

Hasil analisis metode elbow menunjukkan bahwa jumlah cluster optimal adalah 4 dengan nilai total within-cluster sum of squares (WCSS) sebesar 73,679.789. Ini berarti bahwa membagi data menjadi 4 cluster memberikan keseimbangan terbaik antara meminimalkan jarak di dalam cluster dan memaksimalkan jarak antar cluster. 

## Cluster Modeling (K-Means Clustering)

Dengan jumlah cluster yang sudah ditentukan sebanyak 4, kita dapat melanjutkan dengan membangun model clustering menggunakan KMeans. Dalam kode ini, kita melakukan analisis karakteristik cluster setelah melatih model KMeans dengan jumlah cluster yang telah ditetapkan, yaitu 4. Pertama, kita menginisialisasi model KMeans dengan parameter n_clusters=4 dan random_state=0 untuk memastikan hasil yang konsisten. Setelah melatih model dengan data X, kita memperoleh label cluster untuk setiap titik data. 

Fungsi analyze_clusters kemudian digunakan untuk menganalisis karakteristik dari setiap cluster. Fungsi ini mengambil data dari masing-masing cluster berdasarkan label yang diberikan oleh model. Untuk setiap cluster, fungsi ini menghitung rata-rata dari dua fitur: pendapatan tahunan (Annual Income) dan skor belanja (Spending Score). Hasil analisis dicetak untuk setiap cluster menunjukkan rata-rata pendapatan tahunan dan skor belanja yang memberikan wawasan tentang profil pelanggan dalam setiap cluster.

Visualisasi dimulai dengan plot scatter untuk menampilkan data pelanggan yang telah dikelompokkan ke dalam cluster dengan warna berbeda pada setiap cluster berdasarkan pemberian label. Centroid dari setiap cluster digambarkan dengan marker 'X' berwarna merah dan ukuran yang lebih besar. Label ditambahkan pada setiap centroid untuk menandai posisinya. 

Selain itu, kita menambahkan judul serta label pada sumbu X dan Y untuk memberikan konteks pada plot yang menunjukkan distribusi pendapatan tahunan serta skor belanja pelanggan dalam setiap cluster. Setelah visualisasi, nilai centroid untuk setiap cluster ditampilkan. Ini menunjukkan pendapatan tahunan dan skor belanja rata-rata yang mewakili pusat dari masing-masing cluster. Berikut adalah visualisasi penyebaran cluster yang kita dapatkan.

![9](https://github.com/user-attachments/assets/cfb53fd6-52d3-4e8d-8592-614ab190cf75)

Nilai centroid untuk setiap cluster sebagai berikut.

- Centroid 1: pendapatan tahunan $48,260 serta skor belanja 56.48 menunjukkan pelanggan dengan pendapatan menengah dan belanja tinggi.
- Centroid 2: pendapatan tahunan $86,540 serta skor belanja 82.13 menggambarkan pelanggan dengan pendapatan tinggi dan belanja intensif.
- 
