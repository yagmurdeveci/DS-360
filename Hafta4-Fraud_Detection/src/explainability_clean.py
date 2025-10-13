"""SHAP ve LIME için implementasyon"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Modelin kararlarını SHAP ve LIME gibi yöntemlerle açıklamak için sınıf"""
class ModelExplainer:
    def __init__(self, model, X_train, feature_names=None, class_names=None):
        self.model = model
        self.X_train = X_train
        # Feature names eğer verilmemişse otomatik olarak X_train'in kolon isimlerini al
        self.feature_names = feature_names or list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['Normal', 'Fraud'] 
        self.shap_explainer = None
        self.shap_values = None
        self.lime_explainer = None

    # SHAP açıklayıcıyı (Tree veya Kernel) otomatik olarak seçip hazır hale getiriyor.
    def initialize_shap(self, explainer_type='tree'):
        if not SHAP_AVAILABLE:
            logger.warning("SHAP mevcut değil")
            return False
        try:
            # Ağaç tabanlı modeller için hızlı ve optimize TreeExplainer kullan
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Lineer veya karma modeller için KernelExplainer
                # Eğitim verisinden 50 örnek alarak arka plan (background) oluştur
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
            logger.info("SHAP explainer hazır")
            return True
        except Exception as e:
            logger.error(f"SHAP hatası: {e}")
            return False

    # max_samples: Hesaplamayı hızlandırmak için örnek sayısı sınırı
    def compute_shap_values(self, X_test, max_samples=50):
        """SHAP values hesapla"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None, X_test
        
        # Hesaplama süresini azaltmak için test verisinin sadece ilk max_samples kadarını al
        X_sample = X_test.head(max_samples) if hasattr(X_test, 'head') else X_test[:max_samples]

        try:
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            if isinstance(self.shap_values, list):
                return self.shap_values, X_sample
            return self.shap_values, X_sample
        except Exception as e:
            logger.error(f"SHAP computation error: {e}")
            return None, X_sample

    def plot_shap_summary(self, X_test=None):
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP values mevcut değil")
            return 
        plt.figure(figsize=(10, 6))
        # SHAP öznitelik önem grafiğini çiz (özelliklerin etkisini gösterir)
        X_for_plot = X_test if X_test is not None else self.X_train
        shap.summary_plot(
            self.shap_values, X_for_plot,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()

    def initialize_lime(self):
        if not LIME_AVAILABLE:
            logger.warning("LIME mevcut değil")
            return False
        try:
            # LimeTabularExplainer: tablo (tabular) veri üzerinde açıklamalar üretir
            self.lime_explainer = LimeTabularExplainer(
                self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            logger.info("LIME explainer hazır")
            return True
        except Exception as e:
            logger.error(f"LIME hatası: {e}")
            return False

    def explain_instance_lime(self, X_test, instance_idx=0):
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.warning("LIME mevcut değil")
            return None
        
        # Açıklanacak örneği (satırı) test verisinden seç
        # Eğer X_test bir DataFrame ise iloc kullan, değilse doğrudan dizinden al
        instance = X_test.iloc[instance_idx] if hasattr(X_test, 'iloc') else X_test[instance_idx]

        try:
            # LIME açıklaması oluştur
            # instance: açıklanacak tek gözlem(örnek)
            # model.predict_proba: modelin sınıf olasılıklarını döndüren fonksiyon
            # num_features: açıklamada gösterilecek en önemli özellik sayısı
            explanation = self.lime_explainer.explain_instance(
                instance.values if hasattr(instance, 'values') else instance,
                self.model.predict_proba,
                num_features=10
            )

            fig = explanation.as_pyplot_figure(label=1)
            plt.title(f'LIME Explanation - Instance {instance_idx}')
            plt.tight_layout()
            plt.show()

            return explanation
        
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return None

    def permutation_importance(self, X_test, y_test):
        try:
            # Permutasyon yöntemiyle özellik önemini hesapla
            perm_importance = permutation_importance(
                self.model, X_test, y_test,
                n_repeats=5, # Her özellik için tekrar sayısı
                random_state=42
            )

            # En yüksek 10 önemli özelliği sırala
            sorted_idx = np.argsort(perm_importance.importances_mean)[-10:]

            plt.figure(figsize=(10, 6))
            plt.barh(
                [self.feature_names[i] for i in sorted_idx], # Özellik isimleri
                perm_importance.importances_mean[sorted_idx] # Önem değerleri
            )
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (permutation-based)')
            plt.tight_layout()
            plt.show()

            return perm_importance
        except Exception as e:
            logger.error(f"Permutation importance error: {e}")
            return None