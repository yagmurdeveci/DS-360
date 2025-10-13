import os
import pandas as pd
from sklearn.datasets import load_iris

def download_iris_data(output_path='data/raw/iris.csv'):

    # veri dizinlerini oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Iris veri setini yükle
    iris = load_iris(as_frame=True)
    df = iris.frame  

    # Ham veriyi kaydet
    df.to_csv(output_path, index=False)

    print("Iris veri seti indirildi:", output_path)
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")

    return df

if __name__ == "__main__":
    download_iris_data()
