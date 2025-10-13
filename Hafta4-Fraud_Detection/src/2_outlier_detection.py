"""
Outlier Detection (RAW / scale yok)
Yeni Financial Fraud Detection veri seti için Isolation Forest ve LOF
"""

from pathlib import Path
import os, json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

DATA_DIR = Path("data/processed"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# Veri yolu, pipeline tarafından otomatik olarak sağlanacak. 
# Ancak bu scriptin tek başına çalışabilmesi için varsayılan bir yol tanımlayalım.
RAW_PATH = Path("data/raw/Synthetic_Financial_datasets_log.csv")

OUT_CSV  = DATA_DIR / "dataset_with_anomaly_scores_raw.csv"
OUT_META = DATA_DIR / "outlier_meta_raw.json"

def choose_threshold_by_f1(y_true, scores):
    """F1 skorunu maksimize eden eşiği (threshold) bulur."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    thr_choice = float(thr[max(0, min(best_idx-1, len(thr)-1))]) if len(thr) > 0 else 0.0
    return {"threshold": thr_choice, "precision": float(prec[best_idx]),
            "recall": float(rec[best_idx]), "f1": float(f1[best_idx])}

def main():
    """Ana fonksiyon - Outlier detection analizi"""
    if not RAW_PATH.exists():
        print(f"[HATA] Ham veri bulunamadı: {RAW_PATH}")
        print("Lütfen önce 1_download_data.py scriptini çalıştırın.")
        return

    df = pd.read_csv(RAW_PATH)
    
    # Hedef kolon 'isFraud'
    assert "isFraud" in df.columns, "Hedef kolon 'isFraud' bulunamadı."
    
    # Yüksek kardinaliteli kolonları çıkar
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')
    
    # Kategorik kolonları one-hot encode et (bu analiz için basitçe)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Veri setini eğitim ve test olarak ayır (stratified)
    if "split" not in df.columns:
        y_tmp = df["isFraud"].astype(int).values
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=0.30, random_state=42, stratify=y_tmp
        )
        split = np.array(["train"]*len(df), dtype=object); split[idx_test] = "test"
        df["split"] = split

    feature_cols = [c for c in df.columns if c not in ("isFraud", "split")]
    train = df[df["split"]=="train"].reset_index(drop=True)
    test  = df[df["split"]=="test"].reset_index(drop=True)

    X_train = train[feature_cols].values
    X_test  = test[feature_cols].values
    y_test  = test["isFraud"].astype(int).values

    print(f"[OK] Kaynak: {RAW_PATH}")
    print(f"[OK] Train: {X_train.shape} | Test: {X_test.shape} | Test fraud oranı: {y_test.mean():.6f}")

    # --- Isolation Forest ---
    print("\n--- Isolation Forest analizi ---")
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_train)
    if_scores_train = -iso.decision_function(X_train)
    if_scores_test  = -iso.decision_function(X_test)

    if_thr = choose_threshold_by_f1(y_test, if_scores_test)
    if_alarm_test = (if_scores_test >= if_thr["threshold"]).astype(int)

    if_roc = float(roc_auc_score(y_test, if_scores_test))
    if_ap  = float(average_precision_score(y_test, if_scores_test))
    print(f"[IF] ROC-AUC={if_roc:.4f} | PR-AUC(AP)={if_ap:.4f} | "
          f"Eşik≈{if_thr['threshold']:.6f} | P={if_thr['precision']:.3f} R={if_thr['recall']:.3f} F1={if_thr['f1']:.3f} "
          f"| Alarm oranı={if_alarm_test.mean():.4f}")

    # --- LOF (novelty=True) ---
    print("\n--- Local Outlier Factor (LOF) analizi ---")
    lof = LocalOutlierFactor(n_neighbors=35, contamination="auto", novelty=True)
    lof.fit(X_train)
    lof_scores_train = -lof.score_samples(X_train)
    lof_scores_test  = -lof.score_samples(X_test)

    lof_thr = choose_threshold_by_f1(y_test, lof_scores_test)
    lof_alarm_test = (lof_scores_test >= lof_thr["threshold"]).astype(int)

    lof_roc = float(roc_auc_score(y_test, lof_scores_test))
    lof_ap  = float(average_precision_score(y_test, lof_scores_test))
    print(f"[LOF] ROC-AUC={lof_roc:.4f} | PR-AUC(AP)={lof_ap:.4f} | "
          f"Eşik≈{lof_thr['threshold']:.6f} | P={lof_thr['precision']:.3f} R={lof_thr['recall']:.3f} F1={lof_thr['f1']:.3f} "
          f"| Alarm oranı={lof_alarm_test.mean():.4f}")

    # --- Skor/Alarm kolonlarını yaz ---
    df_out = df.copy()
    df_out["if_score"]  = np.nan; df_out["lof_score"] = np.nan
    df_out.loc[df_out["split"]=="train","if_score"]  = if_scores_train
    df_out.loc[df_out["split"]=="train","lof_score"] = lof_scores_train
    df_out.loc[df_out["split"]=="test","if_score"]   = if_scores_test
    df_out.loc[df_out["split"]=="test","lof_score"]  = lof_scores_test

    df_out["if_alarm"]  = 0; df_out["lof_alarm"] = 0
    df_out.loc[df_out["split"]=="test","if_alarm"]  = if_alarm_test
    df_out.loc[df_out["split"]=="test","lof_alarm"] = lof_alarm_test

    df_out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Kaydedildi → {OUT_CSV}")

    meta = {
        "input_file": str(RAW_PATH),
        "output_file": str(OUT_CSV),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "iforest": {"roc_auc": if_roc, "pr_auc_ap": if_ap, **if_thr},
        "lof":     {"roc_auc": lof_roc, "pr_auc_ap": lof_ap, **lof_thr, "n_neighbors": 35},
        "notes": [
            "Ham veri kullanıldı, ölçekleme yapılmadı.",
            "Eşikler PR eğrisinde F1’i maksimize eden noktadan seçildi.",
            "Skorlar train+test için üretildi; testte alarm etiketleri yazıldı."
        ]
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Meta kayıt → {OUT_META}")

if __name__ == "__main__":
    main()