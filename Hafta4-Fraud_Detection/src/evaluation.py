import numpy as np
from sklearn.metrics import(
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

class FraudEvaluator:
    def __init__(self, model=None, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.results = {}
        
    def evaluate_binary_classification(self, x_test, y_true, y_pred_proba=None):
        if y_pred_proba is None and self.model is not None:
            y_pred_proba = self.model.predict_proba(x_test)[:, 1]

        y_pred = (y_pred_proba >= 0.5).astype(int)

        results = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.results = results
        return results