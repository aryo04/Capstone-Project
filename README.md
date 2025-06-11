<p align="center">
  <img src="https://github.com/user-attachments/assets/c55bc528-13c3-4675-a744-c4b6221c93d2"/>
</p>

> **"Deteksi dini, Hidup lebih pasti di DiagnoSmart"**

**DiagnoSmart** adalah platform diagnosis cerdas berbasis web yang memanfaatkan teknologi modern untuk membantu pengguna mengenali potensi penyakit sejak dini. Dengan tampilan elegan, responsif, dan fungsionalitas berbasis AI ringan, DiagnoSmart dirancang untuk menjadi asisten kesehatan pribadi Anda – cepat, mudah, dan aman.

---

## Fitur
| model\_type | Deskripsi Model                                         |
| ----------- | ------------------------------------------------------- |
| `bone`      | Untuk prediksi penyakit tulang dan sendi                |
| `digestive` | Untuk prediksi gangguan sistem pencernaan               |
| `general`   | Model umum untuk berbagai gejala ringan                 |
| `skin`      | Untuk prediksi penyakit kulit seperti ruam atau infeksi |


---

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama.
- **NumPy**: Untuk pemrosesan data input gejala.
- **Pandas**: Untuk manipulasi dan analisis data.
- **TensorFlow**: Framework untuk membangun dan melatih model machine learning.
- **Keras**: API tingkat tinggi untuk merancang dan melatih model neural network.
- **FastAPI**: Framework untuk membangun API.
- **Railway**: Untuk deploy model.
- **Docker**: Untuk containerisasi aplikasi.

---

## 📁 Struktur Folder Penting

```plaintext
models/
├── bone/
│   ├── bone_disease_model.h5
│   ├── bone_label_encoder.pkl
│   └── bone_symptoms.json
├── digestive/
│   ├── digestive_disease_model.h5
│   ├── digestive_label_encoder.pkl
│   └── digestive_symptoms.json
├── general/
│   ├── general_disease_model.h5
│   ├── general_label_encoder.pkl
│   └── general_symptoms.json
├── skin/
│   ├── skin_disease_model.h5
│   ├── skin_label_encoder.pkl
│   └── skin_symptoms.json
```

---
## Cara penggunaan
Model machine learning dideploy menggunakan **Railway** dan dapat digunakan untuk memprediksi penyakit berdasarkan gejala dalam Bahasa Indonesia serta menerima hasil prediksi penyakit berdasarkan model yang dipilih. Setiap model memiliki daftar gejala yang berbeda, untuk melihat gejala yang tersedia untuk masing-masing model, buka file JSON di dalam folder:

```
models/<model_type>/<model_type>_symptoms.json
```

### Menggunakan Postman

1. Buka aplikasi Postman
2. Pilih method `POST`
3. Masukkan URL:
   ```
   https://capstone-project-production-0852.up.railway.app/predict
   ```
4. Buka tab Body → pilih raw → ubah format ke JSON
5. Sebagai contoh untuk bone, masukkan data seperti berikut:
   ```
   {
    "symptoms": ["Nyeri punggung", "Nyeri lutut"],
    "model_type": "bone"
   }
   ```
6. Klik Send   
      

