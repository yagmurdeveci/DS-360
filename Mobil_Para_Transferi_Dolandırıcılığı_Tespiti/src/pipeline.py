"""
Ana Financial Fraud Detection Pipeline
Training, inference ve deployment için end-to-end pipeline
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
import logging
import argparse
import warnings
import importlib.util

# Yerel modülleri dinamik olarak içe aktarmak için yardımcı fonksiyon
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join("src", file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Düzeltilmiş içe aktarmalar
download_data = import_module_from_path("download_data", "1_download_data.py")
outlier_detection = import_module_from_path("outlier_detection", "2_outlier_detection.py")
# Diğer modüller
from preprocessing import FeaturePreprocessor, ImbalanceHandler
from evaluation import FraudEvaluator
from explainability_clean import ModelExplainer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """End-to-end Financial Fraud Detection Pipeline"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        
        self.preprocessor = None
        self.models = {}
        self.evaluators = {}
        self.explainer = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("Financial Fraud Detection Pipeline initialized")
    
    def _load_config(self, config_path):
        """Configuration dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            return {
                'data': {'test_size': 0.3, 'random_state': 42},
                'preprocessing': {'scaling_method': 'robust', 'encoding_method': 'onehot'},
                'models': {'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}}
            }
    
    def _setup_mlflow(self):
        """MLflow setup"""
        mlflow_config = self.config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = mlflow_config.get('experiment_name', 'financial_fraud_detection')
        mlflow.set_experiment(experiment_name)
        if mlflow_config.get('autolog', {}).get('sklearn', True):
            mlflow.sklearn.autolog()
        logger.info(f"MLflow configured - Experiment: {experiment_name}")
    
    def load_data(self, data_path=None, download_with_kagglehub=False):
        """Veri yükleme"""
        if download_with_kagglehub or data_path is None:
            data_path, data = download_data.download_financial_fraud_dataset()
            if data is None:
                raise FileNotFoundError("Dataset indirilemedi.")
        else:
            logger.info(f"Veri yükleniyor: {data_path}")
            data = pd.read_csv(data_path)
        
        self._validate_data(data)
        
        X = data.drop('isFraud', axis=1)
        y = data['isFraud']
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if self.config['data'].get('stratify', True) else None
        )
        
        logger.info(f"Data loaded - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"Class distribution - Train: {np.bincount(self.y_train)}, Test: {np.bincount(self.y_test)}")
    
    def _validate_data(self, data):
        """Data validation"""
        validation_config = self.config.get('data', {}).get('validation', {})
        required_cols = validation_config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("Data validation completed")

    def preprocess_data(self):
        """Data preprocessing"""
        logger.info("Data preprocessing başlatılıyor...")
        
        preprocessing_config = self.config.get('preprocessing', {})
        self.preprocessor = FeaturePreprocessor(
            scaling_method=preprocessing_config.get('scaling_method', 'robust'),
            encoding_method=preprocessing_config.get('encoding_method', 'onehot')
        )
        
        train_data = pd.concat([self.X_train, self.y_train.rename("isFraud")], axis=1)
        train_processed = self.preprocessor.fit_transform(train_data, target_col='isFraud')
        self.X_train_processed = train_processed.drop('isFraud', axis=1)
        self.y_train_processed = train_processed['isFraud']
        
        test_data = pd.concat([self.X_test, self.y_test.rename("isFraud")], axis=1)
        test_processed = self.preprocessor.transform(test_data, target_col='isFraud')
        self.X_test_processed = test_processed.drop('isFraud', axis=1)
        self.y_test_processed = test_processed['isFraud']
        
        logger.info(f"Preprocessing completed - Features: {self.X_train_processed.shape[1]}")
        
        imbalance_config = self.config.get('imbalance', {})
        method = imbalance_config.get('method', 'smote')
        
        if method == 'smote':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_smote(
                self.X_train_processed, self.y_train_processed,
                sampling_strategy=imbalance_config.get('sampling_strategy', 'auto'),
                random_state=imbalance_config.get('random_state', 42)
            )
        else:
            self.X_train_balanced = self.X_train_processed
            self.y_train_balanced = self.y_train_processed
        
        logger.info(f"Class balancing completed - Final training size: {len(self.X_train_balanced)}")
    
    def train_models(self):
        """Model training"""
        logger.info("Model training başlatılıyor...")
        models_config = self.config.get('models', {})
        
        with mlflow.start_run():
            for model_name, model_params in models_config.items():
                logger.info(f"Training {model_name}...")
                
                if model_name == 'random_forest':
                    model = RandomForestClassifier(**model_params)
                elif model_name == 'logistic_regression':
                    model = LogisticRegression(**model_params)
                elif model_name == 'isolation_forest':
                    model = IsolationForest(**model_params)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                if model_name in ['isolation_forest']:
                    model.fit(self.X_train_processed)
                else:
                    model.fit(self.X_train_balanced, self.y_train_balanced)
                
                self.models[model_name] = model
                
                logger.info(f"{model_name} training completed")
            logger.info("All models trained successfully")
    
    def evaluate_models(self):
        """Model evaluation"""
        logger.info("Model evaluation başlatılıyor...")
        evaluation_config = self.config.get('evaluation', {})
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            evaluator = FraudEvaluator(model, model_name)
            
            if model_name in ['isolation_forest']:
                predictions = model.predict(self.X_test_processed)
                y_pred_proba = np.where(predictions == -1, 1, 0)
                results = evaluator.evaluate_binary_classification(
                    self.X_test_processed, self.y_test_processed, y_pred_proba=y_pred_proba
                )
            else:
                results = evaluator.evaluate_binary_classification(
                    self.X_test_processed, self.y_test_processed
                )
            
            self.evaluators[model_name] = evaluator
            
            with mlflow.start_run(nested=True):
                mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
                mlflow.log_metrics({
                    f"{model_name}_roc_auc": results['roc_auc'],
                    f"{model_name}_pr_auc": results['pr_auc'],
                    f"{model_name}_f1_score": results['f1_score'],
                    f"{model_name}_precision": results['precision'],
                    f"{model_name}_recall": results['recall']
                })
                if model_name not in ['isolation_forest']:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            logger.info(f"{model_name} evaluation completed: ROC-AUC={results['roc_auc']:.4f}, PR-AUC={results['pr_auc']:.4f}")
            
            min_roc_auc = evaluation_config.get('min_roc_auc', 0.7)
            min_pr_auc = evaluation_config.get('min_pr_auc', 0.3)
            
            if results['roc_auc'] < min_roc_auc:
                logger.warning(f"{model_name} ROC-AUC ({results['roc_auc']:.4f}) is below threshold ({min_roc_auc})")
            if results['pr_auc'] < min_pr_auc:
                logger.warning(f"{model_name} PR-AUC ({results['pr_auc']:.4f}) is below threshold ({min_pr_auc})")
        
        logger.info("Model evaluation completed")
    
    def explain_models(self, model_name='random_forest'):
        """Model explainability"""
        if 'explainability_clean' not in sys.modules:
            logger.warning("ModelExplainability modülü bulunamadı, açıklama atlanıyor.")
            return None, None
            
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        logger.info(f"Explaining {model_name}...")
        
        explainer = ModelExplainer(
            self.models[model_name],
            self.X_train_processed,
            feature_names=list(self.X_train_processed.columns),
            class_names=['Normal', 'Fraud']
        )
        
        explainer_config = self.config.get('explainability', {}).get('shap', {})
        if explainer.initialize_shap(explainer_type=explainer_config.get('explainer_type', 'tree')):
            shap_values, X_sample = explainer.compute_shap_values(self.X_test_processed, max_samples=explainer_config.get('max_samples', 100))
            if shap_values is not None:
                explainer.plot_shap_summary(X_sample)
        
        logger.info("Model explanation completed")
    
    def save_models(self, save_path="models/"):
        """Model ve preprocessor kaydetme"""
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(save_path, 'preprocessor.pkl'))
        logger.info("Preprocessor saved")
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model saved to {model_path}")
        
        feature_info = {
            'feature_names': list(self.X_train_processed.columns),
            'n_features': len(self.X_train_processed.columns)
        }
        joblib.dump(feature_info, os.path.join(save_path, 'feature_info.pkl'))
        logger.info(f"All models saved to {save_path}")
    
    def run_full_pipeline(self, data_path=None, save_models=True, use_kagglehub=False):
        """Full pipeline execution"""
        logger.info("Full Financial Fraud Detection Pipeline başlatılıyor...")
        
        try:
            self.load_data(data_path, use_kagglehub)
            self.preprocess_data()
            self.train_models()
            self.evaluate_models()
            
            best_model = self._find_best_model()
            if best_model:
                self.explain_models(best_model)
            
            if save_models:
                self.save_models()
            
            logger.info("Full pipeline completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def _find_best_model(self):
        """En iyi modeli bul (ROC-AUC'ye göre)"""
        best_model = None
        best_score = 0
        
        for model_name, evaluator in self.evaluators.items():
            if evaluator.results and 'roc_auc' in evaluator.results:
                roc_auc = evaluator.results['roc_auc']
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model_name
        
        logger.info(f"Best model: {best_model} (ROC-AUC: {best_score:.4f})")
        return best_model or 'random_forest'


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Financial Fraud Detection Pipeline')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--data', help='Data file path (optional, uses KaggleHub if not provided)')
    parser.add_argument('--mode', choices=['train', 'predict', 'explain'], default='train', help='Pipeline mode')
    parser.add_argument('--model', default='random_forest', help='Model name for prediction/explanation')
    parser.add_argument('--load_models', action='store_true', help='Load existing models')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--use_kagglehub', action='store_true', help='Download data with KaggleHub')
    
    args = parser.parse_args()
    
    pipeline = FraudDetectionPipeline(args.config)
    
    if args.mode == 'train':
        success = pipeline.run_full_pipeline(args.data, args.save_models, args.use_kagglehub)
        sys.exit(0 if success else 1)
        
    elif args.mode == 'predict':
        if not args.load_models:
            print("Uyarı: Tahmin modunda --load_models kullanmalısınız.")
            sys.exit(1)
        if not pipeline.load_models():
            sys.exit(1)
            
        print("Tahmin için örnek bir veri girin...")
        # Örnek tahmin yapacak bir arayüz veya kod buraya eklenebilir
        # Örn:
        # sample_data = pd.DataFrame(...)
        # predictions, probabilities = pipeline.predict(sample_data, args.model)
        # print(predictions, probabilities)

    elif args.mode == 'explain':
        if not args.load_models:
            print("Uyarı: Açıklama modunda --load_models kullanmalısınız.")
            sys.exit(1)
        if not pipeline.load_models():
            sys.exit(1)
            
        # Explain için ön işleme yapılmış veriye ihtiyaç var.
        # Bu kısım pipeline'dan ayrı çalışırken dikkat edilmeli.
        # Basitlik için burada örnek sentetik veri kullanabiliriz.
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=len(pipeline.preprocessor.numerical_features) + len(pipeline.preprocessor.categorical_features), random_state=42)
        X_df = pd.DataFrame(X) # Bu kısım gerçek veriye uyarlanmalı.
        
        importance, patterns = pipeline.explain_models(args.model)
        
        print("Top 10 Important Features:")
        if importance:
            for i, (feature, score) in enumerate(list(importance.items())[:10]):
                print(f"{i+1}. {feature}: {score:.4f}")

if __name__ == "__main__":
    main()