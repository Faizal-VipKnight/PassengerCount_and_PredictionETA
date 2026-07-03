# PassengerCount_and_PredictionETA
Kode untuk tugas akhir membuat prediksi ETA(Estimate Time Arrival) dan kombinasi passenger counting dengan yolo dengan hardware raspberry pi 5 dengan kondisi kabin kendaraan terbuka( buggy car) (mirip mobil golf)


Kode ini merupakan kode tugas akhir dengan path yang sengaja dikosongi, 
untuk menjalankan kode ini ini hal yang disiapkan yaitu pertama datasheet harian transportasi buggy car (pada project ini menggunakan EV) , nah dalam datasheet harian ini mirip log data transportasi yang berisi data latitude, longitude, kecepatan, jam, hari, speed, jumlah penumpang, max penumpang, dll sesuai dengan variabel yang anda siapkan

untuk hardware tidak akan dijelasakan detail, yang jelas menggunakan raspberry pi 5 sebagai otak ai untuk fitur passsenger counting, kemudian esp32, modul gps, modul gsm, nah alurnya raspberry pi ngirim data ke esp32, esp32 kirim ke mqtt server, seetlah itu tampil di website

nah untuk menjalankan kode ini, siapkan python
 1. lalu langkah pertama buka gabungin data, 
 gabungindata.py buka ini dan siapkan datasheet anda bentuk csv dan juga parameternya sesuaikan dengan data anda
 2. buka cleaning data(cleaningdata.py), isi path, sesuaikan dengan data anda, dan cleaning data dari hasil yg sudah digabungkan, serta resampling dan bersihkan dari anomali yang sudah anda atur, serta masukan dataroute( rute yg benar yg anda atur) untuk memaksimalkan resampling agar tidak menembus gedung dan tegak lurus ke tengah bidang
 3. kemudian cek dengan visualisasidata.py untuk melihat hasil cleaningnya, ini cuma ambil 5 rute sepertinya
 4. Kemudian siapkan fitur dan parameter untuk kunci model trainingnya di FeatureEngineering.py disini atur path dan nama variabel serta logika logika model nanti, semakin banyak variabel dan parameter lebih bagus
 5. Kemudian training model ainya dengan file trainingdata.py atur dan training semaksimal mungkin, gunakan gpu kalau bisa, dan model sebaiknya dalam output json
 6. terakhir uji dengan model ai lain seperti random forest, multiple linear regresion, silahkan explore dan kembangkan sendiri

Jangan ragu hubungi zalwoko
