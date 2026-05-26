# Human Safety Monitoring System

Sistem monitoring keselamatan manusia berbasis Computer Vision untuk mendeteksi manusia, melakukan tracking pergerakan, serta memantau kepatuhan penggunaan Alat Pelindung Diri (APD/PPE) secara real-time pada area kerja maupun area publik.

---

# Latar Belakang

Kecelakaan kerja masih menjadi salah satu permasalahan utama pada sektor industri, konstruksi, manufaktur, dan area publik. Banyak insiden terjadi akibat kurangnya pengawasan terhadap penggunaan Alat Pelindung Diri (APD) seperti helm keselamatan, safety vest, dan kacamata pelindung.

Sistem pengawasan manual memiliki beberapa keterbatasan:

* Tidak dapat memantau seluruh area secara konsisten
* Bergantung pada operator manusia
* Rentan terhadap human error dan fatigue
* Tidak mampu memberikan analisis real-time secara otomatis

Human Safety Monitoring System hadir sebagai solusi berbasis Artificial Intelligence dan Computer Vision yang mampu melakukan monitoring otomatis melalui kamera CCTV atau video stream secara real-time.

---

# Solusi yang Ditawarkan

Sistem ini dirancang untuk membantu proses monitoring keselamatan kerja secara otomatis menggunakan teknologi deep learning dan multi-object tracking.

Fitur utama sistem:

* Person Detection
* Multi-Object Tracking
* PPE Detection
* Real-time Visualization
* FastAPI Inference Endpoint

Dengan sistem ini, perusahaan atau institusi dapat:

* Memantau kepatuhan penggunaan APD secara otomatis
* Mengurangi risiko kecelakaan kerja
* Mempercepat respons terhadap pelanggaran keselamatan
* Menggunakan infrastruktur CCTV yang sudah tersedia
* Melakukan monitoring real-time tanpa pengawasan manual penuh

---

# Fitur Sistem

## 1. Person Detection

Sistem mendeteksi keberadaan manusia pada gambar maupun video menggunakan model YOLOv8.

Output:

* Bounding box
* Confidence score
* Label object

---

## 2. Person Tracking

Menggunakan ByteTrack untuk menjaga konsistensi tracking ID antar frame video.

Fitur:

* Multi-person tracking
* Tracking ID konsisten
* Support crowded scene

---

## 3. PPE Detection

Mendeteksi penggunaan APD seperti:

* Helmet
* Safety Vest
* Safety Glasses

Sistem dapat mengidentifikasi:

* Menggunakan APD
* Tidak menggunakan APD

---

## 4. Annotated Visualization

Hasil inferensi divisualisasikan secara real-time menggunakan:

* Bounding box
* Tracking ID
* Label PPE
* Confidence score

---

## 5. FastAPI Endpoint

Sistem menyediakan REST API untuk inference.

Endpoint utama:

* `/api/detect`
* `/api/track`
* `/api/ppe`
* `/api/full_pipeline`

---

# Arsitektur Sistem

Pipeline sistem:

Video / CCTV Stream
↓
Frame Extraction (OpenCV)
↓
YOLOv8 Detection
↓
ByteTrack Tracking
↓
PPE Classification
↓
Visualization & API Response
↓
Monitoring Result

---

# Teknologi yang Digunakan

## Computer Vision

* YOLOv8s
* ByteTrack
* PyTorch
* OpenCV

## AI Usage for create code and debugging
* ChatGPT
* Claude Sonnet
* Antigravity (Gemini-Flash(High))

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Tools

* Roboflow
* LabelImg
* TensorBoard / WandB

---

# Dataset

Dataset yang digunakan:

* PPE Dataset (Roboflow)
* Custom Annotated Dataset

Format anotasi:

* YOLO Format (.txt)

---

# Evaluation Metrics

Sistem dievaluasi menggunakan:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* IoU
* MOTA
* FPS Inference

---

# Target Performa

| Metric                   | Target            |
| ------------------------ | ----------------- |
| mAP@0.5 Person Detection | >= 0.85           |
| mAP@0.5 PPE Detection    | >= 0.80           |
| Precision / Recall       | >= 0.82 / >= 0.80 |
| Tracking MOTA            | >= 0.75           |
| Inference Latency        | <= 60ms           |
| API Response Time        | <= 200ms          |

---

# Edge Case Handling

Sistem dirancang untuk menangani:

* Low-light condition
* Partial occlusion
* Crowded scene
* Multiple subjects
* Small object detection

---

# Output Sistem

Sistem menghasilkan:

* Annotated image/video
* Bounding box detection
* Tracking ID
* PPE status
* JSON API response

Contoh response:

```json
{
  "track_id": 1,
  "class": "person",
  "helmet": true,
  "vest": false,
  "confidence": 0.94
}
```

---

# Pengembangan Selanjutnya

Roadmap pengembangan:

* Realtime dashboard
* Telegram/email alert
* People counting
* Edge AI deployment
* ONNX/TensorRT optimization
* Multi-camera monitoring
* Face recognition

---

# Tujuan Pengembangan

Sistem ini dikembangkan sebagai solusi AI berbasis Computer Vision untuk meningkatkan keselamatan kerja dan membantu proses monitoring secara otomatis, real-time, scalable, dan lebih efisien dibandingkan pengawasan manual konvensional.

---

# Author

Human Safety Monitoring System
AI Computer Vision Competition Project
