# Financial Fraud Detection Pipeline

## Proje Amacı
Bu proje, simüle edilmiş mobil para transferi işlemlerinden oluşan bir veri setini kullanarak dolandırıcılık tespiti için uçtan uca bir makine öğrenmesi (ML) pipeline'ı oluşturmayı amaçlar. Proje, veri alımından model değerlendirmesine kadar tüm adımları otomatikleştiren bir MLOps yaklaşımını benimser.

## Kullanılan Dataset
Proje, Kaggle'da bulunan **Financial Fraud Detection Dataset**'ini kullanır. Bu veri seti, PaySim simülatörü ile oluşturulmuştur ve gerçek dünyadaki finansal dolandırıcılık desenlerini yansıtır.

## Proje Yapısı
yeni-fraud-detection-projesi/
├── src/                           # Ana kaynak kodlar
│   ├── 1_download_data.py         # Dataset indirme scripti
│   ├── preprocessing.py           # Veri ön işleme modülü
│   ├── evaluation.py              # Model değerlendirme metrikleri
│   ├── explainability_clean.py    # SHAP ve LIME model açıklaması
│   └── pipeline.py                # Uçtan uca ana pipeline
├── config/                        # Konfigürasyon dosyaları
│   └── config.yaml                # Pipeline parametreleri
├── data/                          # Veri klasörleri
│   └── raw/                       # Ham veri
│   └── processed/                 # İşlenmiş veri
├── models/                        # Eğitilmiş modeller
├── DATASET_STORY.md               # Veri setinin detaylı hikayesi
└── README.md                      # Bu dosya

## Pipeline Mantığı
`src/pipeline.py` scripti, projenin ana omurgasıdır. Tek bir komutla tüm iş akışını yönetir.

1.  **Veri Alımı**: `1_download_data.py` kullanılarak veri seti otomatik olarak indirilir.
2.  **Ön İşleme**: `preprocessing.py` modülü ile kategorik özellikler kodlanır ve bakiye değişimleri gibi yeni özellikler oluşturulur.
3.  **Model Eğitimi**: `config.yaml` dosyasında tanımlanan modeller eğitilir (Random Forest, Logistic Regression vb.).
4.  **Değerlendirme**: `evaluation.py` ile model performansı `ROC-AUC`, `PR-AUC`, `F1 Score` gibi metriklerle ölçülür.
5.  **Açıklanabilirlik**: En iyi modelin tahminleri `explainability_clean.py` kullanılarak açıklanır.
6.  **Model Kaydı**: Eğitilen model ve ön işleme nesnesi `models/` klasörüne kaydedilir.

## Başlangıç
1.  **Bağımlılıkları Kur**: `requirements.txt` dosyasındaki kütüphaneleri yükleyin.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Pipeline'ı Çalıştır**:
    * **Veriyi indirip tüm pipeline'ı çalıştırmak için:**
        ```bash
        python src/pipeline.py --mode train --use_kagglehub --save_models
        ```
    * **Daha önce indirilmiş veri ile çalışmak için:**
        ```bash
        python src/pipeline.py --mode train --save_models
        ```

Bu komutlar, veri indirme işleminden model kaydına kadar tüm adımları sırasıyla çalıştırır.