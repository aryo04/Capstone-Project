<p align="center">
  <img src="https://github.com/user-attachments/assets/c55bc528-13c3-4675-a744-c4b6221c93d2"/>
</p>

> **"Deteksi dini, Hidup lebih pasti di DiagnoSmart"**

**DiagnoSmart** adalah platform diagnosis cerdas berbasis web yang memanfaatkan teknologi modern untuk membantu pengguna mengenali potensi penyakit sejak dini. Dengan tampilan elegan, responsif, dan fungsionalitas berbasis AI ringan, DiagnoSmart dirancang untuk menjadi asisten kesehatan pribadi Anda – cepat, mudah, dan aman.

---

## 🚀 Fitur Model

| `model_type` | Deskripsi Model                                          |
|--------------|----------------------------------------------------------|
| `bone`       | Prediksi penyakit tulang                                 |
| `digestive`  | Prediksi gangguan sistem pencernaan                      |
| `general`    | Prediksi gejala ringan umum                              |
| `skin`       | Prediksi penyakit kulit                                  |

---

## 🛠️ Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **NumPy**: Pemrosesan data input gejala
- **Pandas**: Manipulasi dan analisis data
- **TensorFlow & Keras**: Untuk pembuatan dan pelatihan model ML
- **FastAPI**: Framework API modern
- **Railway**: Platform hosting
- **Docker**: Containerisasi aplikasi

---

## 📁 Struktur Proyek

```plaintext
DiagnoSmartML/
│
├── data/
│   ├── symptom_description.csv
│   ├── symptom_precaution.csv
│   ├── penyakitkulit.csv
│   ├── penyakitpencernaan.csv
│   ├── penyakittulang.csv
│   └── penyakitumum.csv
│
├── models/
│   ├── bone/
│   │   ├── bone_disease_model.h5
│   │   ├── bone_label_encoder.pkl
│   │   └── bone_symptoms.json
│   ├── digestive/
│   │   ├── digestive_disease_model.h5
│   │   ├── digestive_label_encoder.pkl
│   │   └── digestive_symptoms.json
│   ├── general/
│   │   ├── general_disease_model.h5
│   │   ├── general_label_encoder.pkl
│   │   └── general_symptoms.json
│   └── skin/
│       ├── skin_disease_model.h5
│       ├── skin_label_encoder.pkl
│       └── skin_symptoms.json
│
├── notebook/
│   ├── Description_Precaution.ipynb
│   ├── ModelKulit.ipynb
│   ├── ModelPencernaan.ipynb
│   ├── ModelTulang.ipynb
│   └── ModelUmum.ipynb
│
├── Dockerfile
├── main.py
├── requirements.txt
├── utils.py
└── README.md
```

---
## Menjalankan Secara Lokal
1. Clone Repository
  ```
  git clone https://github.com/username/DiagnoSmartML.git
  cd DiagnoSmartML
  ```
2. Membuat dan Aktifkan Virtual Environment
  ```
  python -m venv venv
  # Windows
  venv\Scripts\activate
  # macOS/Linux
  source venv/bin/activate
  ```
3. Install Dependency
  ```
  pip install -r requirements.txt
  ```
4. Menjalankan Program
  ```
  uvicorn main:app --reload
  ```
5. Endpoint
  ```
  http://127.0.0.1:8000/predict
  ```
  Setiap model memiliki daftar gejala yang berbeda, untuk melihat gejala yang tersedia untuk masing-masing model, buka file JSON di dalam        folder:
  ```
  models/<model_type>/<model_type>_symptoms.json
  ```
  Lalu, gunakan method `POST` dengan contoh response `JSON` :
  ```
  {
  "symptoms": ["<isi gejala>", "<isi gejala>"],
  "model_type": "<pilih penyakitnya>"
  }
  ``` 

---

---
## 🌐 Penggunaan API model yang Dideploy
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
      

