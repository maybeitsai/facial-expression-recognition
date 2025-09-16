# Facial Expression Recognition (FER) Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

_Proyek klasifikasi ekspresi wajah menggunakan Convolutional Neural Networks dan Transfer Learning_

</div>

## 📋 Daftar Isi

- [📊 Overview](#-overview)
- [🎯 Tujuan Proyek](#-tujuan-proyek)
- [📁 Struktur Dataset](#-struktur-dataset)
- [🚀 Instalasi dan Setup](#-instalasi-dan-setup)
- [🔧 Penggunaan](#-penggunaan)
- [🏗️ Arsitektur Model](#️-arsitektur-model)
- [📈 Hasil Eksperimen](#-hasil-eksperimen)
- [📊 Evaluasi Performa](#-evaluasi-performa)
- [🔍 Analisis Mendalam](#-analisis-mendalam)
- [💡 Rekomendasi](#-rekomendasi)
- [👥 Kontributor](#-kontributor)
- [📜 Lisensi](#-lisensi)

## 📊 Overview

Proyek ini mengimplementasikan dan membandingkan dua pendekatan **Deep Learning** untuk klasifikasi ekspresi wajah manusia pada dataset FER (Facial Expression Recognition). Sistem ini dapat mengklasifikasikan 7 kategori emosi dasar dari gambar wajah dengan akurasi hingga **59.3%**.

### 🎭 Kategori Emosi

- 😠 **Angry** (Marah)
- 🤢 **Disgust** (Jijik)
- 😨 **Fear** (Takut)
- 😊 **Happy** (Senang)
- 😐 **Neutral** (Netral)
- 😢 **Sad** (Sedih)
- 😲 **Surprise** (Terkejut)

## 🎯 Tujuan Proyek

1. **Mengembangkan model CNN** untuk mengenali ekspresi wajah manusia
2. **Membandingkan performa** antara baseline CNN vs Transfer Learning
3. **Mengevaluasi kinerja** menggunakan classification report dan confusion matrix
4. **Melakukan inference** pada gambar uji untuk validasi praktis
5. **Menganalisis** faktor-faktor yang mempengaruhi performa model

## 📁 Struktur Dataset

```
data/
├── train/                    # Dataset pelatihan (28,709 gambar)
│   ├── angry/               # 4,953 gambar
│   ├── disgust/             # 547 gambar
│   ├── fear/                # 5,121 gambar
│   ├── happy/               # 8,989 gambar
│   ├── neutral/             # 6,198 gambar
│   ├── sad/                 # 6,077 gambar
│   └── surprise/            # 4,002 gambar
└── test/                     # Dataset pengujian (3,589 gambar)
    ├── angry/               # 958 gambar
    ├── disgust/             # 111 gambar
    ├── fear/                # 1,024 gambar
    ├── happy/               # 1,774 gambar
    ├── neutral/             # 1,233 gambar
    ├── sad/                 # 1,247 gambar
    └── surprise/            # 831 gambar
```

### 📊 Karakteristik Dataset

- **Format**: Gambar grayscale 48x48 piksel
- **Total gambar**: 32,298 (28,709 train + 3,589 test)
- **Preprocessing**: Wajah sudah melalui registrasi otomatis
- **Distribusi**: Tidak seimbang (imbalanced), dengan 'disgust' sebagai kelas minoritas

## 🚀 Instalasi dan Setup

### Prerequisites

```bash
Python 3.8+
pip atau conda
```

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Faciel Expression"
```

### 2. Install Dependencies

```bash
pip install tensorflow>=2.8.0
pip install numpy matplotlib scikit-learn
pip install pathlib warnings
```

### 3. Verifikasi Struktur Data

Pastikan folder `data/` dengan struktur yang benar sudah tersedia di direktori proyek.

## 🔧 Penggunaan

### 🚀 Quick Start

1. Buka `main.ipynb` di Jupyter Notebook atau VS Code
2. Jalankan semua cell secara berurutan
3. Model akan dilatih dan disimpan di folder `model/`
4. Hasil evaluasi akan ditampilkan otomatis

### 🎛️ Konfigurasi Parameter

```python
# Parameter yang dapat disesuaikan
IMG_SIZE = (48, 48)           # Ukuran input gambar
TL_IMG_SIZE = (96, 96)        # Ukuran untuk Transfer Learning
BATCH_SIZE = 128              # Ukuran batch
EPOCHS_BASELINE = 30          # Epoch untuk baseline CNN
EPOCHS_TL = 30                # Epoch untuk Transfer Learning
SEED = 2025                   # Seed untuk reproducibility
```

### 📊 Load Model Tersimpan

```python
# Load model yang sudah dilatih
model_baseline = keras.models.load_model('model/fer_model_baseline.h5')
model_mobilenet = keras.models.load_model('model/fer_model_mobilenet.h5')
```

## 🏗️ Arsitektur Model

### 🎯 Model A: Baseline CNN

**Arsitektur Sederhana namun Efektif**

```
Input (48x48x1) → Rescaling (1/255)
├── Conv2D(32) → BatchNorm → MaxPool2D
├── Conv2D(64) → BatchNorm → MaxPool2D
├── Conv2D(128) → BatchNorm → MaxPool2D
├── GlobalAveragePooling2D
├── Dropout(0.3)
├── Dense(256, swish)
└── Dense(7, softmax)
```

**Keunggulan:**

- ✅ Dirancang khusus untuk data FER
- ✅ Efisien secara komputasi
- ✅ Mudah dilatih dan di-debug
- ✅ Menghasilkan akurasi terbaik (59.3%)

### 🔄 Model B: Transfer Learning (MobileNetV2)

**Arsitektur Kompleks dengan Pre-trained Weights**

```
Input (96x96x3) → MobileNetV2(frozen)
├── Conv2D(128) → MaxPool2D
├── GlobalAveragePooling2D
├── Dropout(0.5)
├── Dense(256, swish)
└── Dense(7, softmax)
```

**Proses Pelatihan:**

1. **Phase 1**: Freeze MobileNetV2, latih classifier saja
2. **Phase 2**: Fine-tuning dengan unfreeze lapisan atas

## 📈 Hasil Eksperimen

### 🏆 Perbandingan Performa

| Model              | Akurasi   | Precision | Recall | F1-Score | Parameter |
| ------------------ | --------- | --------- | ------ | -------- | --------- |
| **Baseline CNN**   | **59.3%** | 0.60      | 0.59   | 0.59     | ~380K     |
| **MobileNetV2 TL** | 54.7%     | 0.56      | 0.55   | 0.54     | ~2.3M     |

### 📊 Performa per Kelas (Baseline CNN)

| Emosi        | Precision | Recall   | F1-Score | Support |
| ------------ | --------- | -------- | -------- | ------- |
| Angry        | 0.50      | 0.57     | 0.53     | 958     |
| Disgust      | **0.83**  | 0.32     | 0.46     | 111     |
| Fear         | 0.38      | 0.38     | 0.38     | 1024    |
| **Happy**    | **0.76**  | **0.82** | **0.79** | 1774    |
| Neutral      | 0.60      | 0.61     | 0.61     | 1233    |
| Sad          | 0.55      | 0.57     | 0.56     | 1247    |
| **Surprise** | **0.77**  | **0.73** | **0.75** | 831     |

## 📊 Evaluasi Performa

### 🎯 Temuan Utama

1. **Model Sederhana Menang**: Baseline CNN mengalahkan Transfer Learning
2. **Kelas Terbaik**: Happy (F1: 0.79) dan Surprise (F1: 0.75)
3. **Kelas Tersulit**: Fear (F1: 0.38) dan Disgust (F1: 0.46)
4. **Class Imbalance**: Disgust memiliki precision tinggi (0.83) tapi recall rendah (0.32)

### 📈 Confusion Matrix Insights

- **Happy** dan **Surprise**: Mudah dibedakan karena ciri visual yang distingtif
- **Fear ↔ Sad**: Sering tertukar karena kemiripan ekspresi
- **Angry ↔ Neutral**: Overlap dalam fitur wajah
- **Disgust**: Sulit dideteksi karena jumlah data sangat sedikit

## 🔍 Analisis Mendalam

### 🤔 Mengapa Baseline CNN Lebih Baik?

#### 1. **Domain Mismatch**

- MobileNetV2 dilatih pada **ImageNet** (objek berwarna, tekstur kompleks)
- Dataset FER: **Grayscale, wajah spesifik, 48x48 piksel**
- Fitur ImageNet tidak relevan untuk nuansa ekspresi wajah

#### 2. **Fine-tuning Failure**

- Learning rate terlalu tinggi pada fase fine-tuning
- Model pre-trained "rusak" saat di-update
- Rekomendasi: Gunakan learning rate sangat kecil (1e-5)

#### 3. **Model Complexity vs Data Specificity**

- MobileNetV2 terlalu kompleks untuk dataset homogen
- Baseline CNN memiliki "kapasitas" yang tepat
- Risiko overfitting lebih rendah pada model sederhana

#### 4. **Preprocessing Overhead**

- Konversi grayscale → RGB menambah noise
- Resize 48x48 → 96x96 mengurangi informasi detail
- Baseline CNN bekerja langsung pada data asli

### 📊 Learning Curves Analysis

```python
# Visualisasi training history menunjukkan:
# - Baseline: Konvergensi stabil, no overfitting
# - MobileNetV2: Fluktuasi tinggi, fine-tuning gagal
```

## 💡 Rekomendasi

### 🔧 Immediate Improvements

1. **Fix Fine-tuning Process**

   ```python
   # Gunakan learning rate yang sangat kecil
   model.compile(optimizer=Adam(learning_rate=1e-5), ...)
   ```

2. **Implement Data Augmentation**

   ```python
   datagen = ImageDataGenerator(
       rotation_range=10,
       width_shift_range=0.1,
       height_shift_range=0.1,
       zoom_range=0.1,
       horizontal_flip=True
   )
   ```

3. **Handle Class Imbalance**
   ```python
   # Gunakan class weights
   class_weights = compute_class_weight('balanced',
                                       classes=np.unique(y_train),
                                       y=y_train)
   model.fit(..., class_weight=class_weights)
   ```

### 🚀 Advanced Experiments

4. **Try Different Pre-trained Models**

   - EfficientNetB0 (lebih ringan)
   - VGGFace (spesifik untuk wajah)
   - ResNet50 dengan modifikasi

5. **Ensemble Methods**

   ```python
   # Kombinasi prediksi dari multiple models
   final_pred = (pred_baseline + pred_mobilenet) / 2
   ```

6. **Advanced Architectures**
   - Attention mechanisms
   - Multi-scale feature extraction
   - Facial landmark-guided CNN

### 📈 Future Work

- **Real-time inference** dengan optimasi model
- **Multi-modal approach** (gambar + audio)
- **Cross-dataset validation**
- **Deployment** ke mobile/web application

## 👥 Kontributor

- **Bu Lina** - _Supervisor & Domain Expert_
- **Developer** - _Implementation & Analysis_

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

---

<div align="center">

**🎭 Facial Expression Recognition | Built with ❤️ using TensorFlow**

</div>
