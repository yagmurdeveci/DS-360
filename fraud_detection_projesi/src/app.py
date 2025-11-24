"""
Streamlit Uygulaması: Financial Fraud Detection

Eğitilmiş modeli yükler ve SHAP kullanarak tahminleri açıklar.

NOT: Bu kod, SHAP görselleştirme hatalarını (WaterFall ve Bar Plot) gidermek için 
make_prediction_and_explain fonksiyonunda shap.Explanation nesnesini doğru şekilde oluşturur.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Projenizdeki diğer modülleri (preprocessing, evaluation, explainability_clean) 
# içe aktarmak için 'src' yolu ekleniyor.
# Eğer 'src' klasörü mevcut değilse veya bu modülleri kullanmıyorsanız bu satırı yoruma alabilirsiniz.
# __file__ değişkeni Streamlit'te bazen tanımlı olmayabilir, bu nedenle dosya yolu belirtimi sorun çıkarabilir.
# sys.path.append(os.path.join(os.path.dirname(file), 'src')) 

try:
    # Kullanıcı tanımlı modüller, hata vermesi durumunda yoruma alınabilir.
    from preprocessing import FeaturePreprocessor, ImbalanceHandler
    from evaluation import FraudEvaluator
    from explainability_clean import ModelExplainer
except ImportError:
    pass 

try:
    import shap
    shap.initjs()
except ImportError:
    st.error("SHAP kütüphanesi yüklenemedi. Lütfen pip install shap komutunu çalıştırın.")
    sys.exit()

# Model ve Preprocessor Dosya Yolları
MODEL_PATH = 'models/random_forest_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
FEATURE_INFO_PATH = 'models/feature_info.pkl'

# --- 1. Varlıkları Yükle (Cached) ---
@st.cache_resource
def load_assets():
    """Modelleri ve preprocessor'ı yükler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error("Model dosyaları (random_forest_model.pkl, preprocessor.pkl) bulunamadı.")
        return None, None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        return model, preprocessor, feature_info['feature_names']
    except Exception as e:
        st.error(f"Varlıklar yüklenirken hata oluştu: {e}")
        return None, None, None

# --- 2. Streamlit Başlatma ve Veri Hazırlama ---
st.set_page_config(page_title="Financial Fraud Detector", layout="wide")
st.title("🏦 Mobil Finansal Dolandırıcılık Tespiti")

model, preprocessor, feature_names = load_assets()

if model is None:
    st.stop()

# --- 3. Örnek Tahmin Fonksiyonu ---
def make_prediction_and_explain(raw_data: pd.DataFrame):
    """Veriyi işler, tahmin yapar ve SHAP Explanation nesnesini döndürür."""

    # 1. Veri İşleme
    X_processed = preprocessor.transform(raw_data)
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0
    X_processed = X_processed[feature_names]

    # 2. Tahmin
    proba = model.predict_proba(X_processed)[:, 1][0]
    prediction = model.predict(X_processed)[0]

    # 3. SHAP Açıklaması
    explainer = shap.TreeExplainer(model)
    X_single = X_processed.iloc[[0]]
    shap_values_obj = explainer(X_single)

    # SHAP Değerlerini ve Temel Değeri Çıkarma (Multi-class/Binary Uyumlu)
    if isinstance(shap_values_obj.values, list):
        # Multi-class çıktısı (list of arrays)
        # Genellikle 1. indeks Fraud sınıfını temsil eder
        shap_vals = shap_values_obj.values[1][0] 
        base_val = shap_values_obj.base_values[1]
    elif shap_values_obj.values.ndim == 3 and shap_values_obj.values.shape[-1] == 2:
        # NumPy array çıktısı [1, N_features, 2] şeklinde ise
        shap_vals = shap_values_obj.values[0, :, 1]
        base_val = shap_values_obj.base_values[0, 1]
    else:
        # Binary veya tek sınıf çıktısı
        shap_vals = shap_values_obj.values[0]
        base_val = shap_values_obj.base_values[0]

    # SHAP Explanation nesnesini oluştur (Görselleştirme hatalarını çözer)
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=X_processed.iloc[0].values,
        feature_names=feature_names
    )

    return proba, prediction, X_processed, shap_exp

# --- 4. Arayüz Düzeni ---
st.sidebar.header("İşlem Parametreleri")

amount = st.sidebar.slider("İşlem Tutarı (Amount)", 1000.0, 100000.0, 50000.0, step=100.0)
old_orig = st.sidebar.slider("Gönderici Başlangıç Bakiyesi (oldbalanceOrg)", 0.0, 100000.0, 10000.0)
new_orig = st.sidebar.slider("Gönderici Son Bakiyesi (newbalanceOrig)", 0.0, 100000.0, 0.0)
type_val = st.sidebar.selectbox("İşlem Türü (type)", ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"])

if st.sidebar.button("Tahmin Et ve Açıkla"):
    input_data = pd.DataFrame({
        'step': [100],
        'type': [type_val],
        'amount': [amount],
        'nameOrig': ['C12345'],
        'oldbalanceOrg': [old_orig],
        'newbalanceOrig': [new_orig],
        'nameDest': ['M9876'],
        'oldbalanceDest': [10000],
        'newbalanceDest': [60000],
        'isFlaggedFraud': [0]
    })
    
    try:
        proba, prediction, X_processed, shap_exp = make_prediction_and_explain(input_data)
    except Exception as e:
        st.error(f"Tahmin ve SHAP hesaplama hatası: {e}")
        st.stop()


    # --- Tahmin Sonucu ---
    st.header("1. Tahmin Sonucu")
    col1, col2, col3 = st.columns(3)
    color = "red" if prediction == 1 else "green"
    result_text = f"**{round(proba*100, 2)}%**"
    col1.metric("Dolandırıcılık Olasılığı", result_text, delta_color="off")
    if prediction == 1:
        col2.markdown(f"<h3 style='color:{color};'>🚨 DOLANDIRICILIK (FRAUD)</h3>", unsafe_allow_html=True)
    else:
        col2.markdown(f"<h3 style='color:{color};'>✅ NORMAL İŞLEM</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # --- SHAP Açıklaması ---
    st.header("2. Tahminin Açıklaması (Explainable AI - XAI)")
    st.markdown("Waterfall ve feature importance bar plot gösterilmektedir.")

    try:
        
        # 1. Waterfall Plot (Bireysel Tahmin Açıklaması)
        st.subheader("Bireysel Tahmin Açıklaması (Waterfall Plot)")
        # Plotu göstermek için yeni bir Matplotlib figürü oluştur
        fig_waterfall = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall) # Belleği temizle

        # 2. Bar Plot (Özellik Etki Sıralaması)
        st.subheader("Özellik Etki Sıralaması (Bar Plot)")
        fig_bar = plt.figure(figsize=(10, 5))
        shap.plots.bar(shap_exp, max_display=10, show=False)
        st.pyplot(fig_bar)
        plt.close(fig_bar) # Belleği temizle

        st.info(
            "Kırmızı çubuklar dolandırıcılık olasılığını artırır (pozitif etki).\n"
            "Mavi çubuklar dolandırıcılık olasılığını azaltır (negatif etki)."
        )

    except Exception as e:
        st.error(f"SHAP Görselleştirme Hatası: {e}. Lütfen modelin ve verilerin doğru yüklendiğinden emin olun.")