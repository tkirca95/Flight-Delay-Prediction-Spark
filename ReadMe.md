# ✈️ Büyük Ölçekli Uçuş Gecikmesi Tahmini — Apache Spark MLlib

**Karamanoğlu Mehmetbey Üniversitesi — Fen Bilimleri Enstitüsü**
**Bilgisayar Mühendisliği Anabilim Dalı**
**Büyük Veriye Giriş Dersi — Vize Ödevi (Nisan 2026)**

> Apache Spark MLlib kullanılarak, ABD Ulaştırma Bakanlığı'nın (U.S. DOT) 5.819.079 satırlık 2015 yılı uçuş verisi üzerinde geliştirilen uçtan uca bir ikili sınıflandırma projesi. Lojistik Regresyon, Random Forest ve Gradient Boosted Trees modelleri karşılaştırmalı olarak eğitilmiş; en iyi sonuç **AUC=0,686 ve F1=0,687** değerleriyle GBT modelinden elde edilmiştir.

---

## 📑 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Ana Sonuçlar](#-ana-sonuçlar)
- [Kurulum](#️-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Metodoloji](#-metodoloji)
- [Optimizasyonlar](#-yapılan-optimizasyonlar)
- [Görseller](#-görseller)
- [Atıf](#-atıf)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

### Problem Tanımı

Bir uçuşun kalkış öncesinde bilinebilecek özellikler (ay, gün, havayolu, kalkış/varış havalimanı, planlanan kalkış saati, mesafe) kullanılarak **15 dakikadan fazla rötar yapıp yapmayacağı** tahmin edilmektedir. Problem bir **ikili sınıflandırma (binary classification)** problemi olarak formüle edilmiştir.

### Veri Seti

- **Kaynak:** [Kaggle — usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Boyut:** 5.819.079 satır × 31 kolon (~572 MB)
- **Yıl:** 2015
- **Orjin:** U.S. Department of Transportation (DOT) Bureau of Transportation Statistics

Ayrıntılar için `veri_seti_kaynak.txt` dosyasına bakınız.

### Kullanılan Araçlar

| Kategori | Araç |
|----------|------|
| Dağıtık İşleme | **Apache Spark 3.5 (PySpark API)** |
| Makine Öğrenmesi | **Spark MLlib** (LogisticRegression, RandomForestClassifier, GBTClassifier) |
| Geliştirme Ortamı | **Google Colab** (12.7 GB RAM, 2 CPU) |
| Görselleştirme | Matplotlib, Seaborn |
| Yardımcı Analiz | Pandas, Scikit-Learn, NumPy |

---

## 📊 Ana Sonuçlar

| Model | AUC | Precision | Recall | F1 | Accuracy |
|-------|:---:|:---------:|:------:|:--:|:--------:|
| Logistic Regression | 0,643 | 0,601 | 0,630 | 0,615 | 0,606 |
| Random Forest | 0,672 | 0,616 | 0,663 | 0,638 | 0,624 |
| **GBT (varsayılan eşik)** | **0,686** | 0,630 | 0,649 | 0,639 | **0,634** |
| **GBT (optimum eşik)** | **0,686** | 0,558 | **0,895** | **0,687** | 0,592 |

### Temel Çıkarımlar

- 🏆 **GBT** tablosal verilerde en yüksek başarıyı vermiştir (AUC=0,686)
- 🎯 **Karar eşiği optimizasyonu** F1 skorunu %5 artırmıştır (0,639 → 0,687)
- 🔍 **RF** zamansal örüntülere (kalkış saati), **GBT** mekânsal örüntülere (havalimanı) odaklanmıştır
- 📈 Topluluk yöntemleri, tekil Lojistik Regresyon'u AUC'de 4-5 puan geçmiştir

---

## ⚙️ Kurulum

### Google Colab ile (Önerilen)

1. Kaggle API Token'ınızı alın: Kaggle → Account → "Create New API Token" → `kaggle.json` indirilir
2. `kodlar.ipynb` dosyasını Google Colab'a yükleyin veya [Colab'da açın](https://colab.research.google.com/)
3. Kod çalıştırıldığında `kaggle.json` dosyasını isteyecektir — yükleyin
4. Geri kalan hücreler sırayla çalıştırılabilir

### Yerel Ortamda

```bash
# Bağımlılıkları yükle
pip install pyspark==3.5.0 seaborn matplotlib pandas kaggle scikit-learn numpy

# Kaggle API kurulumu
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Veri setini indir
kaggle datasets download -d usdot/flight-delays
unzip flight-delays.zip -d flight_data

# Scripti çalıştır
python flight_delay_v2.py
```

### Sistem Gereksinimleri

- **Python:** 3.8+ (Colab'da 3.10)
- **Java:** 8 veya 11 (Spark için)
- **RAM:** Minimum 8 GB (tavsiye edilen 12 GB)
- **Disk:** ~2 GB boş alan (veri + ara sonuçlar)

---

## 🚀 Kullanım

Script iki biçimde çalıştırılabilir:

1. **Notebook olarak (`kodlar.ipynb`):** Google Colab veya Jupyter üzerinde hücre hücre çalıştırma
2. **Script olarak (`flight_delay_v2.py`):** Komut satırından doğrudan çalıştırma

Beklenen eğitim süreleri (Google Colab, 2 CPU):

| Aşama | Süre |
|-------|:----:|
| Veri okuma + EDA | ~2 dakika |
| Özellik mühendisliği + undersampling | ~1 dakika |
| Logistic Regression eğitimi | ~2 dakika |
| Random Forest eğitimi | ~15 dakika |
| GBT eğitimi | ~8-25 dakika |
| Değerlendirme + grafikler | ~3 dakika |
| **Toplam** | **~35-50 dakika** |

---

## 📁 Proje Yapısı

```
.
├── kodlar.ipynb                 # Jupyter notebook (Google Colab uyumlu)
├── flight_delay_v2.py           # Python script sürümü
├── ReadMe.md                    # Bu dosya
├── veri_seti_kaynak.txt         # Veri seti kaynak bilgileri
├── rapor.pdf                    # Akademik rapor
└── figures/                     # Eğitim çıktıları (otomatik oluşur)
    ├── fig01_ham_veri_ozeti.png
    ├── fig02_eksik_veri.png
    ├── fig03_sinif_dagilimi_ham.png
    ├── fig04_eda.png                    # EDA (6 panel)
    ├── fig05_time_block_rotar.png
    ├── fig06_undersampling.png
    ├── fig07_train_test_split.png
    ├── fig08_confusion_matrix.png       # 3 model yan yana
    ├── fig09_roc_curve.png              # 3 model ROC
    ├── fig10_threshold_tuning.png       # Karar eşiği optimizasyonu
    ├── fig11_rf_importance.png
    ├── fig12_gbt_importance.png
    ├── fig13_model_karsilastirma.png
    └── fig14_veri_akisi.png
```

---

## 🧪 Metodoloji

### 1. Veri Temizleme

- 31 kolondan **yalnızca 8 kalkış öncesi bilinen** kolon seçildi (veri sızıntısını önlemek için)
- Eksik satırlar `dropna()` ile kaldırıldı → 5,71M satır kaldı
- **Label:** `ARRIVAL_DELAY > 15 dk ? 1 : 0`

### 2. Özellik Mühendisliği

- **Top-N Havalimanı Gruplaması:** 620 benzersiz → Top 50 + "OTHER" (bellek %90 azaldı)
- **Zaman Dilimi (TIME_BLOCK):** Night / Morning / Afternoon / Evening
- **Döngüsel (Cyclical) Kodlama:**
  - `MONTH_sin = sin(2π · (MONTH − 1) / 12)`
  - `DEP_HOUR_sin = sin(2π · DEP_HOUR / 24)` (ve cos versiyonları)

### 3. Sınıf Dengesizliği

Ham veride %82,1 zamanında - %17,9 rötar oranı vardı. **Undersampling** ile 1:1 oranlandı (2,05 milyon satır).

### 4. Çift-Pipeline Mimarisi (Özgün Katkı)

| Pipeline | Kullanım | Aşamalar |
|----------|----------|----------|
| **LR Pipeline** | Lojistik Regresyon | StringIndexer → OneHotEncoder → VectorAssembler → **StandardScaler** |
| **Ağaç Pipeline** | RF + GBT | StringIndexer → VectorAssembler (OHE **YOK**) |

Bu sayede feature vektörü boyutu **~77 → 10**'a düştü; RF/GBT için `maxDepth` artırılabildi.

### 5. Modeller

| Model | numTrees/maxIter | maxDepth | maxBins | Diğer |
|-------|:----------------:|:--------:|:-------:|-------|
| Logistic Regression | maxIter=100 | — | — | regParam=0,01 |
| Random Forest | numTrees=60 | 12 | 128 | subsample=0,8 |
| Gradient Boosted Trees | maxIter=50 | 6 | 128 | stepSize=0,1 |

---

## 🛠️ Yapılan Optimizasyonlar

Toplam 8 kritik iyileştirme uygulandı (raporda **K1–K8** olarak detaylandırılmıştır):

| # | Optimizasyon | Etki |
|---|--------------|------|
| **K1** | Top-50 + OTHER havalimanı gruplaması | Bellek **%90 azaldı** — Colab artık çökmüyor |
| K2 | `cache()/persist()` stratejik kullanımı | **%30-50 hız artışı** |
| K3 | Ağaç modelleri için OHE bypass | RF maxDepth 5→12 çıkabildi |
| K4 | LR için `StandardScaler` | LR AUC **+%2-4** |
| K5 | Üçüncü model olarak **GBTClassifier** | En yüksek AUC (0,686) |
| K6 | Feature importance isim düzeltmesi | Doğru grafikler |
| K7 | Karar eşiği optimizasyonu | F1 **+%5** (0,639 → 0,687) |
| K8 | Tek `groupBy` ile sınıf ve toplam sayımı | Birkaç saniye kazanç |

---

## 🖼️ Görseller

Proje çalıştığında otomatik olarak **14 adet** PNG dosyası üretir. Rapor için kritik olanlar:

- **Şekil 6:** Keşifsel Veri Analizi (6-panel)
- **Şekil 8 & 9:** Confusion Matrix ve ROC Eğrisi (3 model)
- **Şekil 13:** Dört modelin karşılaştırması
- **Şekil 11 & 12:** RF ve GBT özellik önem sıralaması (ilginç fark!)

---

## 📖 Atıf

Bu projeyi atıfta bulunursanız:

```bibtex
@misc{flight_delay_2026,
  title  = {Apache Spark MLlib ile Büyük Ölçekli Uçuş Gecikmesi Sınıflandırması},
  author = {Büyük Veriye Giriş Dersi Vize Ödevi},
  year   = {2026},
  note   = {Karamanoğlu Mehmetbey Üniversitesi, Bilgisayar Mühendisliği}
}
```

### Temel Kaynaklar

- Meng, X. ve ark. (2016). *MLlib: Machine Learning in Apache Spark.* JMLR, 17(34).
- Theodorakopoulos, L. ve ark. (2025). *Optimizing Apache Spark MLlib.* Algorithms, 18(2).
- Tekdoğan, T. & Çakmak, A. (2021). *Benchmarking Spark and Hadoop MapReduce.* ICCBDC.
- Zaharia, M. ve ark. (2012). *Resilient Distributed Datasets.* NSDI'12.

Tam kaynakça için `rapor.pdf` dosyasındaki KAYNAKÇA bölümüne bakınız.

---

## 📜 Lisans

Bu proje akademik bir ödev kapsamında hazırlanmıştır; eğitim amaçlı serbestçe incelenebilir ve referans verilerek kullanılabilir.

Kullanılan veri seti **U.S. Department of Transportation — Bureau of Transportation Statistics** tarafından kamu malı (public domain) olarak yayımlanmıştır.

---

## 🙏 Teşekkür

- **Dr. Öğr. Üyesi Hüseyin ELDEM** — Ders öğretim üyesi, ödev yönetmenliği
- **Apache Spark Topluluğu** — Açık kaynaklı dağıtık hesaplama çerçevesi
- **Kaggle & U.S. DOT** — Açık veri setini sağladıkları için

---

> 💡 *Sorularınız veya önerileriniz için GitHub Issues üzerinden iletişime geçebilirsiniz.*
