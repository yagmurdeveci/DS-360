"""Fraud detection için veri ön işleme araçları"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import(
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging 
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """Financial Detection için kapsamlı feature preprocessing sınıfı"""

    def __init__(self, scaling_method: str = 'standard', encoding_method: str = 'onehot'):
        """  
        Args:
            scaling_method (str): 'standard', 'robust', 'minmax'
            encoding_method (str): 'onehot', 'label', 'ordinal'
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method

        #Scaler seçimi
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method: 'standard', 'robust', veya 'minmax' olmalı")
        
        #Encoder seçimi
        if encoding_method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore')
        elif encoding_method == 'label':
            self.encoder = LabelEncoder()
        elif encoding_method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            raise ValueError("encoding_method: 'onehot', 'label', veya 'ordinal' olmalı")
        
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.is_fitted = False

    def identify_features(self, df: pd.DataFrame):
        """Numerical ve categorical featureları otomatik tespit et"""
        self.numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = df.select_dtypes(include='object').columns.tolist()
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """"Yeni özellikler: bakiye değişimleri ve hata tespiti"""
        df_new = df.copy()
        
        # isFlaggedFraud kolonu varsa çıkar çünkü hedef değişken gibi davranıyor.
        if 'isFlaggedFraud' in df_new.columns:
            df_new = df_new.drop(columns=['isFlaggedFraud'])

        #İşlem türü CASH-IN ve PAYMENT için bakiye değişimini hesapla
        df_new['balance_change_orig'] = df_new['newbalanceOrig'] - df_new['oldbalanceOrg']
        df_new['balance_change_dest'] = df_new['newbalanceDest'] - df_new['oldbalanceDest']

        #Hatalı bakiye durumlarında yeni flag'ler oluştur
        #Transfer sırasında göndericinin ve alıcının bakiye değişimleri tutar ile aynı olmalı
        df_new['orig_balance_error'] = np.where(
            (df_new['type'].isin(['CASH-OUT','TRANSFER'])) &
            (df_new['balance_change_orig'] != -df_new['amount']),
            1,0
        )
        df_new['dest_balance_error'] = np.where(
            (df_new['type'].isin(['CASH-IN','TRANSFER'])) &
            (df_new['balance_change_dest'] != df_new['amount']),
            1,0
        )
    
        #Gereksiz kolonları çıkar
        #nameOrig, nameDest gibi kolonlar yüksek kardinaliteli ve model için kullanışsız
        df_new = df_new.drop(columns=['nameOrig', 'nameDest','step'], errors='ignore')

        logger.info("Feature engineering tamamlandı")
        return df_new

    def fit_transform(self, df: pd.DataFrame, target_col: str=None) ->pd.DataFrame:
        """Fit ve transform işlemlerini birlikte yap"""
        df_processed = df.copy()

        # Hedef sütunu ayır, özellikleri belirle, yeni özellikler oluştur, tekrar özellikleri güncelle
        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col].copy()
            df_processed = df_processed.drop(columns=[target_col])

        self.identify_features(df_processed)
        df_processed = self.create_features(df_processed)
        self.identify_features(df_processed)

        # Sayısal özellikleri ölçeklendir ve logla
        if self.numerical_features:
            df_processed[self.numerical_features] = self.scaler.fit_transform(df_processed[self.numerical_features])
            logger.info(f"Numerical features scaled with {self.scaling_method}")

        if self.categorical_features:
            if self.encoding_method == 'onehot':
                encoded_data = self.encoder.fit_transform(df_processed[self.categorical_features])
                self.encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features).tolist()
                encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_feature_names, index=df_processed.index)
                df_processed = df_processed.drop(columns=self.categorical_features)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            else:
                for col in self.categorical_features:
                    df_processed[col] = self.encoder.fit_transform(df_processed[col])
                
            logger.info(f"Categorical features encoded with {self.encoding_method}")

        self.is_fitted = True

        if target_col and 'target' in locals():
            df_processed[target_col] = target

        return df_processed
    
    # Sadece transform: fit edilmiş preprocessor ile veriyi dönüştür, sayısal ve kategorik özellikleri uygula, hedef sütunu geri ekle
    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Önce fit_transform metodunu çağırın")

        df_processed = df.copy()

        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col].copy()
            df_processed = df_processed.drop(columns=[target_col])

        df_processed = self.create_features(df_processed)
        self.identify_features(df_processed)
        
        #Sayısal özellikleri mevcutsa ölçeklendir
        if self.numerical_features:
            existing_num_features = [f for f in self.numerical_features if f in df_processed.columns]
            if existing_num_features:
                df_processed[existing_num_features] = self.scaler.transform(df_processed[existing_num_features])

        #Kategorik özellikleri encode et:
        #One-hot encoding uygulanırsa yeni sütunlar ekle, değilse ordinal encoding uygula
        if self.categorical_features:
            existing_cat_features = [f for f in self.categorical_features if f in df_processed.columns]
            if existing_cat_features and self.encoding_method == 'onehot' :
                encoded_data = self.encoder.transform(df_processed[existing_cat_features])
                encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_feature_names, index=df_processed.index)
                df_processed = df_processed.drop(columns=existing_cat_features)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            elif existing_cat_features:
                for col in existing_cat_features:
                    df_processed[col] = self.encoder.transform(df_processed[col])

        if target_col and 'target' in locals():
            df_processed[target_col] = target

        return df_processed
            
class ImbalanceHandler:
    @staticmethod
    def apply_smote(X, y, sampling_strategy='auto', random_state=42):
        """SMOTE uygula"""
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"SMOTE uygulandı: {len(X)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled