#!/usr/bin/env python3

import os
import kagglehub
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_financial_fraud_dataset(limit_rows=100_000):
    """
    Financial Fraud Detection dataset indirir ve sadece ilk N satırı alır.
    """
    try:
        logger.info("Financial Fraud Detection dataset indiriliyor...")

        # Dataset'i Kaggle'dan indir
        path = kagglehub.dataset_download("sriharshaeedala/financial-fraud-detection-dataset")
        logger.info(f"Dataset indirildi: {path}")

        # CSV bul
        csv_file_path = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    csv_file_path = os.path.join(root, file)
                    break
            if csv_file_path:
                break

        if csv_file_path:
            logger.info(f"CSV bulundu: {csv_file_path}")

            # Sadece ilk 100k satır yükle
            logger.info(f"İlk {limit_rows:,} satır yükleniyor...")
            df = pd.read_csv(csv_file_path, nrows=limit_rows)

            logger.info(f"Subset data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Class distribution:\n{df['isFraud'].value_counts()}")

            # Kopyalanacak yer
            data_dir = "data/raw"
            os.makedirs(data_dir, exist_ok=True)

            target_path = os.path.join(data_dir, "financial_fraud_100k.csv")
            df.to_csv(target_path, index=False)

            logger.info(f"100k dataset kaydedildi: {target_path}")

            return target_path, df
        else:
            logger.error("CSV dosyası bulunamadı")
            return None, None

    except Exception as e:
        logger.error(f"Dataset download hatası: {e}")
        return None, None


def main():
    """Ana fonksiyon"""
    print("Financial Fraud Detection Dataset Download (100K Version)")
    print("="*60)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    result, df = download_financial_fraud_dataset()

    if result and df is not None:
        print(f"\n Dataset hazır!")
        print(f" Dosya konumu: {result}")

        print(f"\n Dataset Özeti:")
        print(f"   Satır sayısı: {len(df):,}")
        print(f"   Kolon sayısı: {len(df.columns)}")
        print(f"   Normal işlem: {len(df[df['isFraud']==0]):,}")
        print(f"   Fraud işlem: {len(df[df['isFraud']==1]):,}")
        print(f"   Eksik değer: {df.isnull().sum().sum()}")

        print("\n Kullanım örnekleri:")
        print(f"   python src/pipeline.py --mode train --data {result} --save_models")
        print("="*60)
    else:
        print("\n Dataset indirilmedi")

    print("\n Next Steps:")
    print("   1. Dataset indirildi ve 100k versiyonu hazır")
    print("   2. python src/pipeline.py ile modeli eğit")
    print("="*60)

if __name__ == "__main__":
    main()
