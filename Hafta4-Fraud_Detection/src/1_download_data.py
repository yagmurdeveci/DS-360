#!/usr/bin/env python3

import os
import kagglehub
import pandas as pd
import logging 
from pathlib import Path 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_financial_fraud_dataset():
    """
    Financial Fraud Detection dataset indirir.
    """
    try:
        logger.info("Financial Fraud Detection dataset indiriliyor...")

        # Dataset'i Kaggle'dan indir
        path = kagglehub.dataset_download("sriharshaeedala/financial-fraud-detection-dataset")
        logger.info(f"Dataset indirildi: {path}")

        # İndirilen klasördeki tüm dosyaları arayarak CSV dosyasını bul
        csv_file_path = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(root, file)
                    break
            if csv_file_path:
                break

        if csv_file_path:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Class distribution:\n{df['isFraud'].value_counts()}")

            # Proje veri dizinine kopyala
            data_dir = "data/raw"
            os.makedirs(data_dir, exist_ok=True)
            
            target_path = os.path.join(data_dir, os.path.basename(csv_file_path))
            df.to_csv(target_path, index=False)
            logger.info(f"Data kopyalandı: {target_path}")

            return target_path, df
        else:
            logger.error("CSV dosyası bulunamadı")
            return None, None
        
    except Exception as e:
        logger.error(f"Dataset download hatası: {e}")
        return None, None
    
def main():
    """Ana fonksiyon - Financial Fraud dataset indir"""
    print("Financial Fraud Detection Dataset Download")
    print("="*60)
    print("Bu dataset, mobil para transferlerini simüle eder.")
    print("~6.3 milyon işlem")
    print("Farklı işlem tipleri ve bakiye bilgileri içerir.")
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

        print("\n Kullanım örnekleri: ")
        print(f"   # Gerçek data ile training")
        print(f"   python src/pipeline.py --mode train --data {result} --save_models")
        print(f"\n" + "="*60)
    else:
        print("\n Dataset indirilmedi")

    print(f"\n Next Steps:")
    print(f"   1. Dataset indirme tamamlandı ")
    print(f"   2. python src/pipeline.py ile model pipeline'ını çalıştır")
    print(f"\n" + "="*60)

if __name__ == "__main__":
    main()