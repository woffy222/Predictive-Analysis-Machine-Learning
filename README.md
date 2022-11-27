# Laporan Proyek Machine Learning - Mohammad Rafdi
## Domain Proyek
Alpukat merupakan buah yang bergizi, serbaguna, dan lezat, alpukat telah menjadi makanan pokok di banyak rumah di seluruh dunia.
Buah alpukat tumbuh subur di iklim Mediterania, diproduksi di wilayah tersebut, dan kaya akan asam oleat dan serat
Alpukat dianggap sebagai salah satu buah tropis utama, karena mengandung vitamin yang larut dalam lemak yang kurang umum di lain buah-buahan, selain tinggi protein, potasium dan tak jenuh asam lemak. Bubur alpukat mengandung kandungan minyak yang bervariasi, dan banyak  digunakan dalam industri farmasi dan kosmetik, dan di produksi minyak komersial yang mirip dengan minyak zaitun[1][2].

Karena alpukat banyak digunakan untuk bahan pangan dan bahan-bahan dasar untuk memproduksi sesuatu produk. nilai permintaan dan penawaran jika seimbang tidak akan terjadi kenaikan harga namun apabila salah satu nilai permintaan dan penawaran tidak seimbang maka akan terjadi kenaikan atau penurunan harga.
sehingga berdasarkan permasalahan tersebut penulis ingin memprediksi harga buah alpukat sehingga para penyedia alpukat dapat menyeimbangkan nilai penawaran dan permintaan.
## Business Understanding
Petani alpukat akan menjual alpukatnya lebih banyak pada daerah yang memiliki nilai permintaan alpukat yang tinggi dan akan menjual alpukatnya lebih seidkit pada daerah yang memiliki permintaan alpukat yang lebih sedikit.
### Problem Statement
Berdasarkan situasi diatas
- Bagaimana cara preprocessing pada data harga alpukat yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi harga penjualan alpukat ?
### Goals
- Melakukan preprocessing data untuk model machine learning
- Membandingkan model score terbaik untuk memprediksi harga penjualan alpukat 
### Solusion Statement
- Random Forest(RF). Pemilihan metode Random Forest sebagai metode prediksi pada penelitian ini didasari oleh kelebihannya yaitu dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar. Dan kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. Cara kerja Random Forest yakni dengan memanggil fungsi RandomForestRegressor() yang telah diimport dari library scikit-learn[3].
- K-Nearest Neighbor(KNN). Pemilihan metode K-Nearest Neighbor sebagai metode prediksi pada penelitian ini didasari oleh kelebihannya yang mudah dipahami dan diimplementasikan, tangguh terhadap data training sample yang noisy, dan memiliki konsistensi yang kuat. Kekurangannya yakni perlu menentukan parameter k (jumlah tetangga terdekat), sensitif terhadap data outlier. Cara kerja K-Nearest Neighbor yakni dengan memanggil fungsi KNeighborsRegressor() yang telah diimport dari library scikit-learn[4].
- XGBoost (XGB) adalah pendekatan yang ampuh untuk membangun model regresi yang diawasi. Validitas pernyataan ini dapat disimpulkan dengan mengetahui tentang fungsi objektif (XGBoost) dan basis pembelajarnya. Fungsi tujuan berisi fungsi kerugian dan istilah regularisasi. Ini menceritakan tentang perbedaan antara nilai aktual dan nilai prediksi, yaitu seberapa jauh hasil model dari nilai sebenarnya[5].
- Gradient Boosting (GB) adalah salah satu algoritme pembelajaran mesin paling populer untuk kumpulan data tabular. Ini cukup kuat untuk menemukan hubungan nonlinear antara target model dan fitur Anda dan memiliki kegunaan yang hebat yang dapat menangani nilai yang hilang, outlier, dan nilai kategorikal kardinalitas tinggi pada fitur Anda tanpa perlakuan khusus[6].
- LightGBM adalah kerangka peningkatan gradien yang menggunakan algoritma pembelajaran berbasis pohon. Ini dirancang untuk didistribusikan dan efisien[7].
## Data Understanding
Data yang digunakan adalah data yang ada pada kaggle  https://www.kaggle.com/datasets/neuromusic/avocado-prices
keterangan columns pada the dataset:
Date - Tanggal Observasi data

- AveragePrice - Harga rata-rata pada satu buah alpukat dalah mata uang dollar
- type - Konvensional atau organik
- year - Tahun
- Region - Daerah observasi
- Total Volume - Jumlah alpukat yang terjual
- 4046 - Total jumlah alpukat dengan PLU 4046 terjual
- 4225 - Total jumlah alpukat dengan PLU 4225 terjual
- 4770 - Total jumlah alpukat dengan PLU 4770 terjual
##### Note
PLU merupakan Price look-up nomor berisi 4-5 digit untuk mengidentifikasian suatu produk, berguna untuk memudahkan proses check-out dan juga memudahkan inventory control.

![readdataset](https://user-images.githubusercontent.com/118952537/204123039-444a1317-cc1d-448b-b80e-75bd6e7ac33e.png)
![info](https://user-images.githubusercontent.com/118952537/204123147-4a7e8a06-ecb0-4311-b0d3-d2fda0b310ed.png)

#### Hasil analisis
- Sampel data berisi 18249 kolom dan 12 baris
- DataType terdiri dari datetime, float, obj dan int
- nilai minimum untuk harga 1 buah alpukat adalah 0.44 dollar
- nilai maximum untuk harga 1 buah alpukat adalah 3.25 dollar

#### Visualisasi
Grafik distribusi

![grafik distribusi](https://user-images.githubusercontent.com/118952537/204123329-94350848-9800-493a-acd8-c60f1aef98ec.png)


Korelasi Matrix

![korelasi matix](https://user-images.githubusercontent.com/118952537/204123355-c865c45b-b0c9-4adf-a332-06ad0875fc44.png)

## Data Preparation
 Teknik data preparation yang digunakan adalah: 
 - MinMaxScaller() dimana ketika menggunakan teknik ini kita harus menghilangkan kolum yang bernilai data type object. merupakan proses scalling yang fungsinya data numeric akan tahan terhadap pencilan data / outliers. MinMaxScaller ini mentransformasi / mengubah data numeric menjadi data numeric yang memiliki rentang 0 - 1
 - TrainTestSplit() untuk membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.
## Modeling

Pada tahap ini mengembangkan model machine learning dengan lima algoritma, yakni XGBoost, K-Nearest Neighbor, Random Forest, Gradient Boosting, LightGBM. Langkah selanjutnya yakni mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Langkah pertama dalam proses modeling yakni menyiapkan sebuah DataFrame baru untuk menampung berapa nilai mae-nya yang berfungsi pada proses analisis model.
![korelasi matix](https://user-images.githubusercontent.com/118952537/204123834-0976dd91-b998-473f-be4d-da716c8ba051.png)
 
 #### Random Forest (RF)
 ##### Keuntungan 
 - RF dapat menyelesaikan kedua jenis masalah yaitu klasifikasi dan regresi dan melakukan estimasi yang layak di kedua sisi.
 - Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang.
 ##### Kekurangan 
 - seperti pendekatan kotak hitam untuk pemodel statistik, kami hanya memiliki sedikit kendali atas apa yang dilakukan model tersebut.
 
 #### XGBoost
 ##### Keuntungan
 - Efektif dengan kumpulan data besar. Algoritme pohon seperti XGBoost dan Random Forest tidak memerlukan fitur yang dinormalisasi dan bekerja dengan baik jika datanya nonlinier, nonmonotonik, atau dengan kluster terpisah.
 ##### Kekurangan
 - Tidak bekerja dengan baik apabila data tidak terstruktur
 
 #### LGBM
 ##### Keuntungan
 - Kecepatan latihan lebih cepat dan efisiensi lebih tinggi
 - Menggunakan memori yang relatif kecil
 ##### Kekurangan
 - Light GBM peka terhadap overfitting dan karenanya dapat dengan mudah membuat data kecil overfitting
 
#### KNN
##### Keuntungan
- Pemodelan KNN tidak termasuk periode pelatihan karena data itu sendiri adalah model yang akan menjadi acuan untuk prediksi masa depan dan karenanya sangat efisien waktu dalam hal improvisasi untuk pemodelan acak pada data yang tersedia.
##### kekurangan 
- Tidak bekerja dengan baik dengan kumpulan data besar.

#### Gradient Boosting
##### Keuntungan
- dapat mengoptimalkan berbagai fungsi kerugian dan menyediakan beberapa opsi penyetelan parameter hiper yang membuat fungsi tersebut sangat fleksibel
##### Kekurangan
- Model Peningkatan Gradien akan terus ditingkatkan untuk meminimalkan semua kesalahan. Ini dapat terlalu menekankan outlier dan menyebabkan overfitting.

## Evaluation
Metrik evaluasi yang digunakan adalah:
### Mean Squared Error (MSE)
Mean Squared Error (MSE) adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai prediksi. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada prediksi. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual dan bisa dijadikan untuk perhitungan prediksi di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi.
### Root  Mean Squared Error (RMSE)
Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.
Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.
### R2 Score
R squared merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen. Semakin mendekati angka satu, model yang dikeluarkan oleh regresi tersebut akan semakin baik.
Jika kita perhatikan rumus R squared dibawah sangat dipengaruhi oleh nilai Y prediksi atau nilai Y dari hasil rumus dengan nilai Y aktual. Kenyataan yang sering muncul adalah nilai R squared akan semakin membaik (nilainya akan terus mendekati nilai 1) jika kita menambah variabel. Semakin banyak jumlah variabel yang menentukan nilai Y prediksi, maka nilai SSR akan semakin besar yang berakibat pada besarnya nilai R squared.

## Kesimpulan
![hasil akhir](https://user-images.githubusercontent.com/118952537/204124448-ed5c7c52-886b-4df1-a6be-b61825a52a14.png)

Setelah melalui berbagai tahapan evaluasi dan membandingkan ke-5 algoritma yang digunakan yakni, LGBM, XGBoost, KNN, Gradient Boosting dan Random Forest. Nilai skor terbesar adalah algoritma Random Forest dengan nilai 81% disusul oleh LGBM dengan score 76% kemudian KNN dengan skor 73% dan Gradient Boosting dan XGboost dengan skor 64%

## Referensi
[1]	N. A. Ford and A. G. Liu, “The Forgotten Fruit: A Case for Consuming Avocado Within the Traditional Mediterranean Diet,” Front Nutr, vol. 7, May 2020, doi: 10.3389/fnut.2020.00078.

[2]	P. F. Duarte, M. A. Chaves, C. D. Borges, and C. R. B. Mendonça, “Abacate: Características, benefícios à saúde e aplicações,” Ciencia Rural, vol. 46, no. 4, pp. 747–754, Apr. 2016, doi: 10.1590/0103-8478cr20141516.

[3]	V. Y. Kulkarni, “Random Forest Classifiers :A Survey and Future Research Directions.”

[4]	G. Guo, H. Wang, D. A. Bell, Y. Bi, D. Bell, and K. Greer, “KNN Model-Based Approach in Classification TiO2 Photocatalysis and its application in environmental and energy transition View project Smoking and health View project KNN Model-Based Approach in Classification,” 2004. [Online]. Available: https://www.researchgate.net/publication/2948052

[5]	T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Aug. 2016, vol. 13-17-August-2016, pp. 785–794. doi: 10.1145/2939672.2939785.

[6]	A. Natekin and A. Knoll, “Gradient boosting machines, a tutorial,” Front Neurorobot, vol. 7, no. DEC, 2013, doi: 10.3389/fnbot.2013.00021.

[7]	G. Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” [Online]. Available: https://github.com/Microsoft/LightGBM
