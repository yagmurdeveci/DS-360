import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(model_type='random_forest'):

    # İşlenmiş veriyi yükle
    df = pd.read_csv('data/processed/iris_processed.csv')

    # X ve y'yi ayır
    feature_cols = [c for c in df.columns if c not in ['species','target','target_encoded']]
    X = df[feature_cols]
    y = df['target_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model seç
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    # Model eğit
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)

    # Model kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path)

    # Metrikleri kaydet
    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open('models/features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f" Model eğitildi: {model_type}, accuracy={accuracy:.4f}")
    print(" Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, metrics

if __name__ == "__main__":
    train_model('random_forest')
    train_model('logistic_regression')
