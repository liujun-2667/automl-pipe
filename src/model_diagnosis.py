"""
模型对比与诊断模块
- 模型对比表
- 分类任务诊断: 混淆矩阵、ROC曲线、PR曲线、分类报告、SHAP
- 回归任务诊断: 残差图、残差分布、QQ图
- 过拟合检测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ModelDiagnostician:
    """模型诊断器"""

    def __init__(self, task_type: str = 'binary', random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """划分训练集和测试集用于诊断"""
        if self.task_type in ['binary', 'multiclass']:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size,
                stratify=y if self.task_type != 'regression' else None,
                random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=self.random_state
            )
        return self

    def evaluate_model(self, model) -> Dict:
        """评估模型并生成预测结果"""
        model.fit(self.X_train, self.y_train)

        self.y_pred = model.predict(self.X_test)

        if hasattr(model, 'predict_proba') and self.task_type in ['binary', 'multiclass']:
            self.y_proba = model.predict_proba(self.X_test)
        else:
            self.y_proba = None

        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'score_gap': train_score - test_score,
            'is_overfitting': (train_score - test_score) > 0.1,
        }

        if self.task_type in ['binary', 'multiclass']:
            metrics.update(self._classification_metrics())
        else:
            metrics.update(self._regression_metrics())

        return metrics

    def _classification_metrics(self) -> Dict:
        """分类任务指标"""
        metrics = {
            'accuracy': (self.y_pred == self.y_test).mean(),
        }

        try:
            report = classification_report(
                self.y_test, self.y_pred,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report
        except:
            metrics['classification_report'] = {}

        if self.y_proba is not None:
            if self.task_type == 'binary':
                try:
                    auc = roc_auc_score(self.y_test, self.y_proba[:, 1])
                    metrics['roc_auc'] = auc

                    fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:, 1])
                    metrics['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                    }

                    precision, recall, _ = precision_recall_curve(
                        self.y_test, self.y_proba[:, 1]
                    )
                    ap = average_precision_score(self.y_test, self.y_proba[:, 1])
                    metrics['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'average_precision': ap,
                    }
                except:
                    pass
            elif self.task_type == 'multiclass':
                try:
                    auc = roc_auc_score(
                        self.y_test, self.y_proba,
                        multi_class='ovr',
                        average='macro'
                    )
                    metrics['roc_auc_ovr'] = auc
                except:
                    pass

        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            classes = sorted(pd.unique(self.y_test))
            metrics['classes'] = [str(c) for c in classes]
        except:
            metrics['confusion_matrix'] = []
            metrics['classes'] = []

        return metrics

    def _regression_metrics(self) -> Dict:
        """回归任务指标"""
        y_true = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test
        y_pred = self.y_pred

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

        residuals = y_true - y_pred

        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'residuals': residuals.tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
        }

    def get_shap_values(self, model, X: pd.DataFrame, max_samples: int = 200) -> Optional[Dict]:
        """计算SHAP值（如果shap库可用）"""
        try:
            import shap

            sample_size = min(max_samples, len(X))
            X_sample = X.sample(n=sample_size, random_state=self.random_state)

            try:
                explainer = shap.TreeExplainer(model)
            except:
                try:
                    explainer = shap.LinearExplainer(model, X_sample)
                except:
                    try:
                        explainer = shap.KernelExplainer(
                            model.predict,
                            shap.sample(X_sample, 50)
                        )
                    except:
                        return None

            try:
                shap_values = explainer.shap_values(X_sample)
            except:
                return None

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            feature_importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            top_features = X_sample.columns[top_indices].tolist()
            top_importances = feature_importance[top_indices]

            return {
                'shap_values': shap_values,
                'feature_names': X_sample.columns.tolist(),
                'top_features': top_features,
                'top_importances': top_importances.tolist(),
                'X_sample': X_sample,
                'explainer': explainer,
            }
        except ImportError:
            return None
        except Exception as e:
            return None

    def check_overfitting(self, train_score: float, test_score: float, threshold: float = 0.1) -> bool:
        """检查是否过拟合"""
        return (train_score - test_score) > threshold

    def get_overfitting_suggestions(self) -> List[str]:
        """获取过拟合建议"""
        return [
            "增加正则化强度 (如增加L1/L2惩罚项)",
            "减少特征数量 (移除不重要的特征)",
            "增加训练数据量",
            "降低模型复杂度 (如减小树深度、减少树数量)",
            "使用早停 (early stopping)",
            "增加交叉验证折数",
        ]


def compare_models(results_df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """对比所有模型"""
    if results_df.empty:
        return results_df

    df = results_df.copy()

    if task_type == 'regression':
        metric_col = 'cv_mean'
        df['rank'] = df[metric_col].rank(ascending=True)
    else:
        metric_col = 'cv_mean'
        df['rank'] = df[metric_col].rank(ascending=False)

    df = df.sort_values('rank').reset_index(drop=True)

    return df


def format_model_comparison(results_df: pd.DataFrame, task_type: str) -> List[Dict]:
    """格式化模型对比表数据"""
    if results_df.empty:
        return []

    rows = []
    for _, row in results_df.iterrows():
        formatted_params = ', '.join([
            f"{k}={v}" for k, v in row.get('best_params', {}).items()
        ][:5])

        rows.append({
            'model_name': row['model_name'],
            'best_params': formatted_params,
            'cv_mean': round(row['cv_mean'], 4),
            'cv_std': round(row['cv_std'], 4),
            'train_time': round(row.get('train_time', 0), 2),
        })

    return rows
