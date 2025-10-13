import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_iris_data(input_path='data/raw/iris.csv', output_path='data/processed/iris_processed.csv'):

    # Veriyi yükle
    df = pd.read_csv(input_path)

    df_clean = df.copy()

    # Kategorik hedefi encode et
    label_col = 'target' if 'target' in df.columns else 'species'
    df['target_encoded'] = LabelEncoder().fit_transform(df[label_col])

    # Çıktıyı kaydet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Veri temizlendi ve kaydedildi:", output_path)
    print(f"Boyut: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    features = [c for c in df.columns if c not in [label_col, 'target_encoded']]
    print(f"Model özellikleri: {features}")

    return df, features

if __name__ == "__main__":
    clean_iris_data()
