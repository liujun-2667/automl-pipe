"""
模型可解释性分析模块
- 局部解释: SHAP瀑布图、LIME特征权重、ICE单样本边际效应
- 全局解释: SHAP全局重要性、排列重要性、PDP偏依赖图、特征稳定性分析
- 对抗性解释检测: 微小扰动后解释稳定性
- HTML报告导出: 汇总所有分析结果的交互式报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import kendalltau
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.base import clone
import warnings
import json
import base64
from datetime import datetime

warnings.filterwarnings('ignore')


class LocalInterpreter:
    """局部解释器：对单条样本进行多方法解释"""

    def __init__(self, model, X: pd.DataFrame, task_type: str = 'binary',
                 feature_names: Optional[List[str]] = None, random_state: int = 42):
        self.model = model
        self.X = X
        self.task_type = task_type
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self._shap_explainer = None
        self._shap_values = None
        self._lime_explainer = None
        self._init_explainers()

    def _init_explainers(self):
        try:
            import shap
            X_sample = self.X.sample(n=min(100, len(self.X)), random_state=self.random_state)
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                try:
                    self._shap_explainer = shap.LinearExplainer(self.model, X_sample)
                except Exception:
                    try:
                        predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
                        self._shap_explainer = shap.KernelExplainer(
                            predict_fn, shap.sample(X_sample, 50, random_state=self.random_state)
                        )
                    except Exception:
                        self._shap_explainer = None
        except ImportError:
            self._shap_explainer = None

        try:
            from lime.lime_tabular import LimeTabularExplainer
            self._lime_explainer = LimeTabularExplainer(
                self.X.values,
                feature_names=self.feature_names,
                class_names=['class_0', 'class_1'] if self.task_type == 'binary' else None,
                mode='classification' if self.task_type in ['binary', 'multiclass'] else 'regression',
                random_state=self.random_state,
            )
        except ImportError:
            self._lime_explainer = None

    def explain_shap(self, sample_idx: int) -> Dict:
        if self._shap_explainer is None:
            return {'error': 'SHAP explainer not available'}

        try:
            X_sample = self.X.iloc[[sample_idx]]
            shap_values = self._shap_explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values

            base_value = None
            if hasattr(self._shap_explainer, 'expected_value'):
                base_value = self._shap_explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            feature_contribs = []
            for i, feat in enumerate(self.feature_names):
                feature_contribs.append({
                    'feature': feat,
                    'value': float(X_sample.iloc[0, i]),
                    'shap_value': float(shap_values[i]),
                    'abs_shap': float(abs(shap_values[i])),
                })

            feature_contribs.sort(key=lambda x: x['abs_shap'], reverse=True)

            return {
                'base_value': float(base_value) if base_value is not None else 0.0,
                'feature_contributions': feature_contribs,
                'top_features': [f['feature'] for f in feature_contribs[:5]],
                'prediction': float(self.model.predict(X_sample)[0]),
            }
        except Exception as e:
            return {'error': str(e)}

    def explain_lime(self, sample_idx: int, num_features: int = 10) -> Dict:
        if self._lime_explainer is None:
            return {'error': 'LIME explainer not available'}

        try:
            X_sample = self.X.iloc[sample_idx].values

            if self.task_type in ['binary', 'multiclass']:
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            exp = self._lime_explainer.explain_instance(
                X_sample, predict_fn, num_features=num_features
            )

            feature_weights = []
            for feat, weight in exp.as_list():
                feature_weights.append({
                    'feature': feat,
                    'weight': float(weight),
                    'abs_weight': float(abs(weight)),
                })

            feature_weights.sort(key=lambda x: x['abs_weight'], reverse=True)

            top_features_clean = []
            for fw in feature_weights[:5]:
                raw = fw['feature']
                for fn in self.feature_names:
                    if fn in raw:
                        top_features_clean.append(fn)
                        break
                else:
                    top_features_clean.append(raw)

            return {
                'feature_weights': feature_weights,
                'top_features': top_features_clean,
                'prediction': float(self.model.predict(self.X.iloc[[sample_idx]])[0]),
            }
        except Exception as e:
            return {'error': str(e)}

    def explain_ice(self, sample_idx: int, features: Optional[List[str]] = None,
                    grid_resolution: int = 30) -> Dict:
        try:
            if features is None:
                numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
                features = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols

            sample = self.X.iloc[[sample_idx]].copy()
            ice_results = {}

            for feat in features:
                if feat not in self.X.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(self.X[feat]):
                    continue

                feat_min = self.X[feat].min()
                feat_max = self.X[feat].max()
                feat_std = self.X[feat].std()

                if feat_std == 0 or np.isnan(feat_std):
                    continue

                grid_values = np.linspace(feat_min, feat_max, grid_resolution)

                predictions = []
                for val in grid_values:
                    sample_modified = sample.copy()
                    sample_modified[feat] = val
                    if hasattr(self.model, 'predict_proba') and self.task_type in ['binary', 'multiclass']:
                        pred = self.model.predict_proba(sample_modified)
                        pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
                    else:
                        pred = self.model.predict(sample_modified)
                    predictions.append(float(pred[0]))

                original_val = float(sample[feat].iloc[0])
                original_idx = np.argmin(np.abs(grid_values - original_val))
                original_prediction = predictions[original_idx] if original_idx < len(predictions) else predictions[-1]
                mean_prediction = float(np.mean(predictions))
                local_contribution = original_prediction - mean_prediction

                ice_results[feat] = {
                    'grid_values': grid_values.tolist(),
                    'predictions': predictions,
                    'original_value': original_val,
                    'original_prediction': original_prediction,
                    'mean_prediction': mean_prediction,
                    'local_contribution': local_contribution,
                    'local_contribution_abs': float(abs(local_contribution)),
                    'feature_std': float(feat_std),
                    'marginal_effect': float(max(predictions) - min(predictions)),
                }

            sorted_feats = sorted(
                ice_results.items(),
                key=lambda x: x[1]['local_contribution_abs'],
                reverse=True
            )
            top_features = [f for f, _ in sorted_feats[:5]]

            return {
                'ice_curves': ice_results,
                'top_features': top_features,
                'sample_original_values': {f: float(sample[f].iloc[0]) for f in features if f in sample.columns},
            }
        except Exception as e:
            return {'error': str(e)}

    def compute_consistency_score(self, shap_result: Dict, lime_result: Dict,
                                  ice_result: Dict) -> Dict:
        shap_top = shap_result.get('top_features', [])
        lime_top = lime_result.get('top_features', [])
        ice_top = ice_result.get('top_features', [])

        def pairwise_kendall(rank1: List[str], rank2: List[str]) -> float:
            all_feats = list(set(rank1 + rank2))
            if len(all_feats) < 2:
                return 1.0

            r1 = [rank1.index(f) if f in rank1 else len(rank1) for f in all_feats]
            r2 = [rank2.index(f) if f in rank2 else len(rank2) for f in all_feats]

            try:
                tau, _ = kendalltau(r1, r2)
                return float(tau) if not np.isnan(tau) else 0.0
            except Exception:
                return 0.0

        tau_shap_lime = pairwise_kendall(shap_top, lime_top)
        tau_shap_ice = pairwise_kendall(shap_top, ice_top)
        tau_lime_ice = pairwise_kendall(lime_top, ice_top)

        mean_tau = float(np.mean([tau_shap_lime, tau_shap_ice, tau_lime_ice]))
        is_conflict = mean_tau < 0.6

        return {
            'kendall_shap_lime': tau_shap_lime,
            'kendall_shap_ice': tau_shap_ice,
            'kendall_lime_ice': tau_lime_ice,
            'mean_consistency': mean_tau,
            'is_conflict': is_conflict,
            'label': '解释冲突' if is_conflict else '解释一致',
        }


class GlobalInterpreter:
    """全局解释器：对整个验证集进行聚合分析"""

    def __init__(self, model, X: pd.DataFrame, y: pd.Series,
                 task_type: str = 'binary', feature_names: Optional[List[str]] = None,
                 random_state: int = 42):
        self.model = model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def shap_global_importance(self, max_samples: int = 200) -> Dict:
        try:
            import shap

            X_sample = self.X.sample(n=min(max_samples, len(self.X)), random_state=self.random_state)

            try:
                explainer = shap.TreeExplainer(self.model)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(self.model, X_sample)
                except Exception:
                    try:
                        predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
                        explainer = shap.KernelExplainer(
                            predict_fn, shap.sample(X_sample, 50, random_state=self.random_state)
                        )
                    except Exception:
                        return {'error': 'Could not create SHAP explainer'}

            try:
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                return {'error': 'Could not compute SHAP values'}

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_std = np.std(shap_values, axis=0)

            importance_list = []
            for i, feat in enumerate(self.feature_names):
                if i < len(mean_abs_shap):
                    importance_list.append({
                        'feature': feat,
                        'mean_abs_shap': float(mean_abs_shap[i]),
                        'shap_std': float(shap_std[i]),
                    })

            importance_list.sort(key=lambda x: x['mean_abs_shap'], reverse=True)

            unstable_threshold = np.percentile([x['shap_std'] for x in importance_list], 75)
            for item in importance_list:
                item['is_unstable'] = item['shap_std'] > unstable_threshold

            return {
                'importances': importance_list,
                'top_features': [x['feature'] for x in importance_list[:10]],
                'shap_values_matrix': shap_values.tolist(),
                'X_sample': X_sample,
            }
        except ImportError:
            return {'error': 'shap library not installed'}
        except Exception as e:
            return {'error': str(e)}

    def permutation_importance(self, n_repeats: int = 10) -> Dict:
        try:
            from sklearn.metrics import get_scorer
            if self.task_type in ['binary', 'multiclass']:
                scoring = 'roc_auc' if self.task_type == 'binary' else 'f1_macro'
            else:
                scoring = 'r2'

            result = permutation_importance(
                self.model, self.X, self.y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=1,
            )

            importance_list = []
            for i, feat in enumerate(self.feature_names):
                importance_list.append({
                    'feature': feat,
                    'importance_mean': float(result.importances_mean[i]),
                    'importance_std': float(result.importances_std[i]),
                })

            importance_list.sort(key=lambda x: x['importance_mean'], reverse=True)

            return {
                'importances': importance_list,
                'top_features': [x['feature'] for x in importance_list[:10]],
                'scoring': scoring,
            }
        except Exception as e:
            return {'error': str(e)}

    def compute_pdp(self, feature: str, grid_resolution: int = 30) -> Dict:
        try:
            if feature not in self.X.columns:
                return {'error': f'Feature {feature} not found'}

            if not pd.api.types.is_numeric_dtype(self.X[feature]):
                return {'error': f'Feature {feature} is not numeric'}

            pdp_result = partial_dependence(
                self.model, self.X, features=[feature],
                grid_resolution=grid_resolution,
                method='brute',
            )

            return {
                'feature': feature,
                'grid_values': pdp_result['values'][0].tolist(),
                'partial_dependence': pdp_result['average'][0].tolist(),
                'individual': pdp_result['individual'].tolist() if 'individual' in pdp_result else [],
            }
        except Exception as e:
            return {'error': str(e)}

    def compute_pdp_2d(self, feature1: str, feature2: str,
                       grid_resolution: int = 20) -> Dict:
        try:
            for feat in [feature1, feature2]:
                if feat not in self.X.columns:
                    return {'error': f'Feature {feat} not found'}
                if not pd.api.types.is_numeric_dtype(self.X[feat]):
                    return {'error': f'Feature {feat} is not numeric'}

            pdp_result = partial_dependence(
                self.model, self.X, features=[feature1, feature2],
                grid_resolution=grid_resolution,
                method='brute',
            )

            grid1 = pdp_result['values'][0]
            grid2 = pdp_result['values'][1]
            z_values = pdp_result['average'][0]

            return {
                'feature1': feature1,
                'feature2': feature2,
                'grid_x': grid1.tolist(),
                'grid_y': grid2.tolist(),
                'z_values': z_values.tolist(),
            }
        except Exception as e:
            return {'error': str(e)}

    def feature_stability_analysis(self, shap_result: Optional[Dict] = None) -> Dict:
        if shap_result is None or 'error' in shap_result:
            shap_result = self.shap_global_importance()

        if 'error' in shap_result:
            return {'error': shap_result['error']}

        importances = shap_result.get('importances', [])

        if not importances:
            return {'error': 'No SHAP data available'}

        std_values = [x['shap_std'] for x in importances]
        mean_std = float(np.mean(std_values))
        median_std = float(np.median(std_values))

        unstable_features = [x for x in importances if x.get('is_unstable', False)]
        stable_features = [x for x in importances if not x.get('is_unstable', False)]

        return {
            'stability_summary': {
                'mean_shap_std': mean_std,
                'median_shap_std': median_std,
                'n_unstable': len(unstable_features),
                'n_stable': len(stable_features),
                'total_features': len(importances),
            },
            'unstable_features': sorted(unstable_features, key=lambda x: x['shap_std'], reverse=True),
            'stable_features': sorted(stable_features, key=lambda x: x['mean_abs_shap'], reverse=True),
        }


class AdversarialExplainer:
    """对抗性解释检测：检测样本解释的稳定性"""

    def __init__(self, model, X: pd.DataFrame, task_type: str = 'binary',
                 feature_names: Optional[List[str]] = None,
                 column_types: Optional[Dict] = None, random_state: int = 42):
        self.model = model
        self.X = X
        self.task_type = task_type
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.column_types = column_types or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.local_interpreter = LocalInterpreter(model, X, task_type, self.feature_names, random_state)

    def _generate_perturbed_sample(self, sample_idx: int) -> pd.DataFrame:
        sample = self.X.iloc[[sample_idx]].copy()

        for feat in self.feature_names:
            if feat not in sample.columns:
                continue

            col_type = self.column_types.get(feat, '')
            series = self.X[feat]

            if pd.api.types.is_numeric_dtype(series) or col_type == 'numeric':
                feat_std = series.std()
                if feat_std > 0 and not np.isnan(feat_std):
                    perturbation = self.rng.normal(0, 0.5 * feat_std)
                    sample[feat] = sample[feat].values[0] + perturbation
            else:
                value_counts = series.value_counts()
                if len(value_counts) >= 2:
                    current_val = sample[feat].values[0]
                    other_vals = [v for v in value_counts.index if v != current_val]
                    if other_vals:
                        sample[feat] = other_vals[0]

        return sample

    def detect_sensitivity(self, sample_idx: int) -> Dict:
        try:
            original_shap = self.local_interpreter.explain_shap(sample_idx)
            original_lime = self.local_interpreter.explain_lime(sample_idx)
            original_ice = self.local_interpreter.explain_ice(sample_idx)

            if 'error' in original_shap or 'error' in original_lime or 'error' in original_ice:
                return {'error': 'One or more explainers failed on original sample'}

            perturbed_sample = self._generate_perturbed_sample(sample_idx)
            X_with_perturbed = pd.concat([self.X, perturbed_sample], ignore_index=True)
            perturbed_idx = len(X_with_perturbed) - 1

            perturbed_interpreter = LocalInterpreter(
                self.model, X_with_perturbed, self.task_type, self.feature_names, self.random_state
            )

            perturbed_shap = perturbed_interpreter.explain_shap(perturbed_idx)
            perturbed_lime = perturbed_interpreter.explain_lime(perturbed_idx)
            perturbed_ice = perturbed_interpreter.explain_ice(perturbed_idx)

            if 'error' in perturbed_shap or 'error' in perturbed_lime or 'error' in perturbed_ice:
                return {'error': 'One or more explainers failed on perturbed sample'}

            return self._compute_sensitivity_result(
                sample_idx,
                original_shap, original_lime, original_ice,
                perturbed_shap, perturbed_lime, perturbed_ice,
                perturbed_sample,
            )
        except Exception as e:
            return {'error': str(e)}

    def _compute_sensitivity_result(
        self, sample_idx: int,
        original_shap: Dict, original_lime: Dict, original_ice: Dict,
        perturbed_shap: Dict, perturbed_lime: Dict, perturbed_ice: Dict,
        perturbed_sample: pd.DataFrame,
    ) -> Dict:
        def top3_kendall(rank1: List[str], rank2: List[str]) -> float:
            all_feats = list(set(rank1[:3] + rank2[:3]))
            if len(all_feats) < 2:
                return 1.0

            r1 = [rank1.index(f) if f in rank1 else len(rank1) for f in all_feats]
            r2 = [rank2.index(f) if f in rank2 else len(rank2) for f in all_feats]

            try:
                tau, _ = kendalltau(r1, r2)
                return float(tau) if not np.isnan(tau) else 0.0
            except Exception:
                return 0.0

        tau_shap = top3_kendall(original_shap['top_features'], perturbed_shap['top_features'])
        tau_lime = top3_kendall(original_lime['top_features'], perturbed_lime['top_features'])
        tau_ice = top3_kendall(original_ice['top_features'], perturbed_ice['top_features'])

        mean_tau = float(np.mean([tau_shap, tau_lime, tau_ice]))
        is_sensitive = mean_tau < 0.5

        original_pred = float(self.model.predict(self.X.iloc[[sample_idx]])[0])
        perturbed_pred = float(self.model.predict(perturbed_sample)[0])
        prediction_change = abs(original_pred - perturbed_pred)

        return {
            'sample_idx': sample_idx,
            'is_sensitive': is_sensitive,
            'label': '解释敏感样本' if is_sensitive else '解释稳定样本',
            'mean_kendall_tau': mean_tau,
            'shap_tau': tau_shap,
            'lime_tau': tau_lime,
            'ice_tau': tau_ice,
            'original_prediction': original_pred,
            'perturbed_prediction': perturbed_pred,
            'prediction_change': float(prediction_change),
            'original_top3': {
                'shap': original_shap['top_features'][:3],
                'lime': original_lime['top_features'][:3],
                'ice': original_ice['top_features'][:3],
            },
            'perturbed_top3': {
                'shap': perturbed_shap['top_features'][:3],
                'lime': perturbed_lime['top_features'][:3],
                'ice': perturbed_ice['top_features'][:3],
            },
        }

    def batch_detect(self, sample_indices: Optional[List[int]] = None, n_samples: int = 10) -> Dict:
        if sample_indices is None:
            if len(self.X) <= n_samples:
                sample_indices = list(range(len(self.X)))
            else:
                sample_indices = self.rng.choice(len(self.X), size=n_samples, replace=False).tolist()

        if not sample_indices:
            return {
                'total_samples': 0, 'n_sensitive': 0, 'n_stable': 0,
                'adversarial_pass_rate': 0.0, 'sample_results': [], 'sensitive_samples': [],
            }

        n = len(sample_indices)

        perturbed_samples_list = []
        for idx in sample_indices:
            perturbed_samples_list.append(self._generate_perturbed_sample(idx))

        all_perturbed_X = pd.concat(perturbed_samples_list, ignore_index=True)
        X_combined = pd.concat([self.X, all_perturbed_X], ignore_index=True)

        combined_interpreter = LocalInterpreter(
            self.model, X_combined, self.task_type, self.feature_names, self.random_state
        )

        len_X = len(self.X)

        results = []
        for i, idx in enumerate(sample_indices):
            perturbed_idx_in_combined = len_X + i

            try:
                original_shap = self.local_interpreter.explain_shap(idx)
                original_lime = self.local_interpreter.explain_lime(idx)
                original_ice = self.local_interpreter.explain_ice(idx)

                if 'error' in original_shap or 'error' in original_lime or 'error' in original_ice:
                    continue

                perturbed_shap = combined_interpreter.explain_shap(perturbed_idx_in_combined)
                perturbed_lime = combined_interpreter.explain_lime(perturbed_idx_in_combined)
                perturbed_ice = combined_interpreter.explain_ice(perturbed_idx_in_combined)

                if 'error' in perturbed_shap or 'error' in perturbed_lime or 'error' in perturbed_ice:
                    continue

                perturbed_sample = perturbed_samples_list[i]

                result = self._compute_sensitivity_result(
                    idx,
                    original_shap, original_lime, original_ice,
                    perturbed_shap, perturbed_lime, perturbed_ice,
                    perturbed_sample,
                )
                results.append(result)
            except Exception:
                continue

        n_sensitive = sum(1 for r in results if r.get('is_sensitive', False))
        pass_rate = float(1 - n_sensitive / len(results)) if results else 0.0

        return {
            'total_samples': len(results),
            'n_sensitive': n_sensitive,
            'n_stable': len(results) - n_sensitive,
            'adversarial_pass_rate': pass_rate,
            'sample_results': results,
            'sensitive_samples': [r for r in results if r.get('is_sensitive', False)],
        }


class InterpretabilityReportExporter:
    """可解释性报告导出器：生成交互式HTML报告"""

    def __init__(self, model, X: pd.DataFrame, y: pd.Series,
                 task_type: str = 'binary', model_name: str = 'Unknown',
                 feature_names: Optional[List[str]] = None,
                 column_types: Optional[Dict] = None, random_state: int = 42):
        self.model = model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.column_types = column_types or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.local_interpreter = LocalInterpreter(model, X, task_type, self.feature_names, random_state)
        self.global_interpreter = GlobalInterpreter(model, X, y, task_type, self.feature_names, random_state)
        self.adversarial_explainer = AdversarialExplainer(
            model, X, task_type, self.feature_names, column_types, random_state
        )

    def _generate_plotly_figure(self, fig) -> str:
        try:
            import plotly.io as pio
            return pio.to_html(fig, include_plotlyjs=False, full_html=False)
        except ImportError:
            return '<p>Plotly not available</p>'

    def generate_executive_summary(self, global_shap: Dict, adversarial_result: Dict) -> Dict:
        n_features = len(self.feature_names)
        n_samples = len(self.X)

        if 'importances' in global_shap:
            mean_consistency_list = []
            n_sample_checks = min(5, len(self.X))
            check_indices = self.rng.choice(len(self.X), size=n_sample_checks, replace=False).tolist()

            for idx in check_indices:
                shap_res = self.local_interpreter.explain_shap(idx)
                lime_res = self.local_interpreter.explain_lime(idx)
                ice_res = self.local_interpreter.explain_ice(idx)
                if 'error' not in shap_res and 'error' not in lime_res and 'error' not in ice_res:
                    consistency = self.local_interpreter.compute_consistency_score(shap_res, lime_res, ice_res)
                    mean_consistency_list.append(consistency['mean_consistency'])

            overall_consistency = float(np.mean(mean_consistency_list)) if mean_consistency_list else 0.5
        else:
            overall_consistency = 0.5

        adversarial_pass_rate = adversarial_result.get('adversarial_pass_rate', 0.5)

        score = (overall_consistency * 0.5 + adversarial_pass_rate * 0.5) * 100

        if score >= 75:
            grade = '高'
            suggestion = '模型解释可信度高，可放心用于关键决策场景。'
        elif score >= 50:
            grade = '中'
            suggestion = '模型解释可信度中等，建议对敏感样本进行人工复核。'
        else:
            grade = '低'
            suggestion = '模型解释可信度低，建议重新训练模型或增加训练数据。'

        return {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'n_features': n_features,
            'n_samples': n_samples,
            'overall_consistency': overall_consistency,
            'adversarial_pass_rate': adversarial_pass_rate,
            'credibility_score': float(score),
            'credibility_grade': grade,
            'suggestion': suggestion,
        }

    def export_html_report(self, output_path: str) -> Dict:
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            return {'error': 'plotly is required for HTML report export'}

        global_shap = self.global_interpreter.shap_global_importance()
        global_perm = self.global_interpreter.permutation_importance()
        feature_stability = self.global_interpreter.feature_stability_analysis(global_shap)

        if len(self.X) >= 10:
            sample_indices = self.rng.choice(len(self.X), size=min(10, len(self.X)), replace=False).tolist()
        else:
            sample_indices = list(range(len(self.X)))
        adversarial_result = self.adversarial_explainer.batch_detect(sample_indices, n_samples=len(sample_indices))

        summary = self.generate_executive_summary(global_shap, adversarial_result)

        n_local_samples = min(3, len(self.X))
        local_sample_indices = self.rng.choice(len(self.X), size=n_local_samples, replace=False).tolist()
        local_results = []
        for idx in local_sample_indices:
            shap_res = self.local_interpreter.explain_shap(idx)
            lime_res = self.local_interpreter.explain_lime(idx)
            ice_res = self.local_interpreter.explain_ice(idx)
            consistency = None
            if 'error' not in shap_res and 'error' not in lime_res and 'error' not in ice_res:
                consistency = self.local_interpreter.compute_consistency_score(shap_res, lime_res, ice_res)
            local_results.append({
                'sample_idx': idx,
                'shap': shap_res,
                'lime': lime_res,
                'ice': ice_res,
                'consistency': consistency,
            })

        pdp_feature = None
        if 'top_features' in global_shap and global_shap['top_features']:
            pdp_feature = global_shap['top_features'][0]
        pdp_result = self.global_interpreter.compute_pdp(pdp_feature) if pdp_feature else {'error': 'No feature'}

        html_parts = []

        plotly_cdn = '''<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>'''

        html_parts.append(f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型可解释性分析报告 - {summary['model_name']}</title>
    {plotly_cdn}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; border-bottom: 2px solid #2980b9; padding-bottom: 8px; margin-top: 40px; }}
        h3 {{ color: #3498db; }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            margin: 20px 0;
        }}
        .summary-card h2 {{ color: white; border-color: white; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{ font-size: 14px; color: #666; margin-bottom: 4px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .grade-high {{ color: #27ae60; font-size: 32px; font-weight: bold; }}
        .grade-mid {{ color: #f39c12; font-size: 32px; font-weight: bold; }}
        .grade-low {{ color: #e74c3c; font-size: 32px; font-weight: bold; }}
        .badge-conflict {{
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .badge-ok {{
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .badge-sensitive {{
            display: inline-block;
            background: #e67e22;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
        }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .plot-container {{ margin: 20px 0; }}
        .suggestion-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 16px;
            border-radius: 4px;
            margin: 16px 0;
        }}
        .unstable-tag {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
''')

        html_parts.append(f'''
<h1>🔬 模型可解释性分析报告</h1>
<p style="color: #666;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="summary-card">
    <h2>📋 执行摘要</h2>
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-label">模型类型</div>
            <div class="metric-value">{summary['model_name']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">任务类型</div>
            <div class="metric-value">{'分类' if summary['task_type'] in ['binary', 'multiclass'] else '回归'}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">特征数量</div>
            <div class="metric-value">{summary['n_features']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">样本数量</div>
            <div class="metric-value">{summary['n_samples']:,}</div>
        </div>
    </div>
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-label">整体一致性评分</div>
            <div class="metric-value">{summary['overall_consistency']:.3f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">对抗性检测通过率</div>
            <div class="metric-value">{summary['adversarial_pass_rate']*100:.1f}%</div>
        </div>
    </div>
    <h3>🏆 模型可信度评级</h3>
    <p>综合得分: <strong>{summary['credibility_score']:.1f}/100</strong></p>
    <p class="grade-{ 'high' if summary['credibility_grade']=='高' else ('mid' if summary['credibility_grade']=='中' else 'low') }">
        {summary['credibility_grade']}
    </p>
    <div class="suggestion-box">
        💡 <strong>使用建议:</strong> {summary['suggestion']}
    </div>
</div>
''')

        html_parts.append('<h2>🌐 全局解释分析</h2>')

        if 'importances' in global_shap:
            shap_fig = go.Figure()
            top_n = min(15, len(global_shap['importances']))
            top_importances = global_shap['importances'][:top_n]
            shap_fig.add_trace(go.Bar(
                x=[x['mean_abs_shap'] for x in reversed(top_importances)],
                y=[x['feature'] for x in reversed(top_importances)],
                orientation='h',
                marker_color=[
                    '#e74c3c' if x.get('is_unstable', False) else '#3498db'
                    for x in reversed(top_importances)
                ],
                name='mean(|SHAP|)',
            ))
            shap_fig.update_layout(
                title=f'Top {top_n} 特征 SHAP 全局重要性（红色=不稳定）',
                xaxis_title='mean(|SHAP value|)',
                yaxis_title='特征',
                height=500,
            )
            html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(shap_fig)}</div>')

        if 'importances' in global_perm:
            perm_fig = go.Figure()
            top_n = min(15, len(global_perm['importances']))
            top_importances = global_perm['importances'][:top_n]
            perm_fig.add_trace(go.Bar(
                x=[x['importance_mean'] for x in reversed(top_importances)],
                y=[x['feature'] for x in reversed(top_importances)],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=[x['importance_std'] for x in reversed(top_importances)],
                    visible=True,
                ),
                marker_color='#9b59b6',
                name='Permutation Importance',
            ))
            perm_fig.update_layout(
                title=f'Top {top_n} 特征排列重要性',
                xaxis_title=f"重要性得分 ({global_perm.get('scoring', '')})",
                yaxis_title='特征',
                height=500,
            )
            html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(perm_fig)}</div>')

        if 'stability_summary' in feature_stability:
            stab = feature_stability['stability_summary']
            html_parts.append(f'''
<h3>📊 特征稳定性分析</h3>
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">SHAP标准差均值</div>
        <div class="metric-value">{stab['mean_shap_std']:.4f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">SHAP标准差中位数</div>
        <div class="metric-value">{stab['median_shap_std']:.4f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">不稳定特征数</div>
        <div class="metric-value"><span class="unstable-tag">{stab['n_unstable']}</span>/{stab['total_features']}</div>
    </div>
</div>
''')
            if feature_stability.get('unstable_features'):
                html_parts.append('<h4>⚠️ 不稳定特征列表（对不同样本贡献差异大）</h4><table><tr><th>特征</th><th>mean(|SHAP|)</th><th>SHAP Std</th></tr>')
                for f in feature_stability['unstable_features'][:10]:
                    html_parts.append(f"<tr><td>{f['feature']}</td><td>{f['mean_abs_shap']:.4f}</td><td class='unstable-tag'>{f['shap_std']:.4f}</td></tr>")
                html_parts.append('</table>')

        if 'partial_dependence' in pdp_result:
            pdp_fig = go.Figure()
            pdp_fig.add_trace(go.Scatter(
                x=pdp_result['grid_values'],
                y=pdp_result['partial_dependence'],
                mode='lines+markers',
                name='Partial Dependence',
                line=dict(color='#e74c3c', width=3),
            ))
            pdp_fig.update_layout(
                title=f'PDP偏依赖图 - {pdp_result["feature"]}',
                xaxis_title=pdp_result['feature'],
                yaxis_title='预测值（偏依赖）',
                height=400,
            )
            html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(pdp_fig)}</div>')

        html_parts.append('<h2>🔍 局部解释示例</h2>')
        for i, lr in enumerate(local_results):
            html_parts.append(f'<h3>样本 #{lr["sample_idx"]} (示例 {i+1})</h3>')

            if lr.get('consistency'):
                cons = lr['consistency']
                badge_class = 'badge-conflict' if cons['is_conflict'] else 'badge-ok'
                html_parts.append(f'''
<p>
    一致性评分: <strong>{cons['mean_consistency']:.3f}</strong>
    <span class="{badge_class}">{cons['label']}</span>
</p>
<table>
    <tr><th>方法对</th><th>Kendall τ</th></tr>
    <tr><td>SHAP vs LIME</td><td>{cons['kendall_shap_lime']:.3f}</td></tr>
    <tr><td>SHAP vs ICE</td><td>{cons['kendall_shap_ice']:.3f}</td></tr>
    <tr><td>LIME vs ICE</td><td>{cons['kendall_lime_ice']:.3f}</td></tr>
</table>
''')

            if 'feature_contributions' in lr['shap']:
                contribs = lr['shap']['feature_contributions'][:10]
                shap_waterfall = go.Figure()
                cum_val = lr['shap'].get('base_value', 0)
                shap_waterfall.add_trace(go.Waterfall(
                    orientation='v',
                    measure=['absolute'] + ['relative'] * len(contribs) + ['total'],
                    x=['E[f(x)]'] + [f"{c['feature']}={c['value']:.2f}" for c in contribs] + ['f(x)'],
                    y=[cum_val] + [c['shap_value'] for c in contribs] + [sum([cum_val] + [c['shap_value'] for c in contribs])],
                    connector={'line': {'color': 'rgb(100,100,100)'}},
                ))
                shap_waterfall.update_layout(
                    title=f'SHAP瀑布图 - 样本 #{lr["sample_idx"]}',
                    height=500,
                )
                html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(shap_waterfall)}</div>')

            if 'feature_weights' in lr['lime']:
                weights = lr['lime']['feature_weights'][:10]
                lime_fig = go.Figure()
                lime_fig.add_trace(go.Bar(
                    x=[w['weight'] for w in reversed(weights)],
                    y=[w['feature'] for w in reversed(weights)],
                    orientation='h',
                    marker_color=['#e74c3c' if w['weight'] < 0 else '#27ae60' for w in reversed(weights)],
                ))
                lime_fig.update_layout(
                    title=f'LIME特征权重 - 样本 #{lr["sample_idx"]}',
                    xaxis_title='特征权重',
                    height=450,
                )
                html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(lime_fig)}</div>')

            if 'ice_curves' in lr['ice']:
                ice_fig = go.Figure()
                for feat, ice_data in list(lr['ice']['ice_curves'].items())[:3]:
                    ice_fig.add_trace(go.Scatter(
                        x=ice_data['grid_values'],
                        y=ice_data['predictions'],
                        mode='lines',
                        name=feat,
                    ))
                    ice_fig.add_vline(
                        x=ice_data['original_value'],
                        line_dash='dash',
                        annotation_text=f'{feat}原值',
                    )
                ice_fig.update_layout(
                    title=f'ICE边际效应曲线 - 样本 #{lr["sample_idx"]}',
                    xaxis_title='特征值',
                    yaxis_title='预测值',
                    height=450,
                )
                html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(ice_fig)}</div>')

        html_parts.append('<h2>🛡️ 对抗性解释检测</h2>')
        adv = adversarial_result
        html_parts.append(f'''
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">检测样本数</div>
        <div class="metric-value">{adv['total_samples']}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">解释敏感样本数</div>
        <div class="metric-value">{adv['n_sensitive']}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">通过率</div>
        <div class="metric-value">{adv['adversarial_pass_rate']*100:.1f}%</div>
    </div>
</div>
''')

        if adv.get('sample_results'):
            html_parts.append('<h3>详细检测结果</h3><table><tr><th>样本</th><th>状态</th><th>Mean τ</th><th>SHAP τ</th><th>LIME τ</th><th>ICE τ</th><th>预测变化</th></tr>')
            for r in adv['sample_results']:
                badge = '<span class="badge-sensitive">敏感</span>' if r['is_sensitive'] else '<span class="badge-ok">稳定</span>'
                html_parts.append(
                    f"<tr><td>#{r['sample_idx']}</td><td>{badge}</td>"
                    f"<td>{r['mean_kendall_tau']:.3f}</td>"
                    f"<td>{r['shap_tau']:.3f}</td>"
                    f"<td>{r['lime_tau']:.3f}</td>"
                    f"<td>{r['ice_tau']:.3f}</td>"
                    f"<td>{r['prediction_change']:.4f}</td></tr>"
                )
            html_parts.append('</table>')

        html_parts.append(f'''
<h2>🏁 结论</h2>
<div class="summary-card">
    <h3>最终可信度评级: <span class="grade-{ 'high' if summary['credibility_grade']=='高' else ('mid' if summary['credibility_grade']=='中' else 'low') }">{summary['credibility_grade']}</span></h3>
    <div class="suggestion-box">
        💡 {summary['suggestion']}
    </div>
</div>
</body>
</html>
''')

        html_content = '\n'.join(html_parts)

        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return {
            'output_path': output_path,
            'summary': summary,
            'global_shap': {k: v for k, v in global_shap.items() if k not in ['shap_values_matrix', 'X_sample']},
            'feature_stability': feature_stability,
            'adversarial_result': {k: v for k, v in adversarial_result.items() if k != 'sample_results'},
        }


class ModelInterpretabilityAnalyzer:
    """模型可解释性分析主类：整合所有功能"""

    def __init__(self, model, X: pd.DataFrame, y: pd.Series,
                 task_type: str = 'binary', model_name: str = 'Unknown',
                 feature_names: Optional[List[str]] = None,
                 column_types: Optional[Dict] = None, random_state: int = 42):
        self.model = model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.column_types = column_types or {}
        self.random_state = random_state

        self.local = LocalInterpreter(model, X, task_type, self.feature_names, random_state)
        self.global_ = GlobalInterpreter(model, X, y, task_type, self.feature_names, random_state)
        self.adversarial = AdversarialExplainer(
            model, X, task_type, self.feature_names, self.column_types, random_state
        )
        self.reporter = InterpretabilityReportExporter(
            model, X, y, task_type, model_name, self.feature_names, self.column_types, random_state
        )

    def run_full_analysis(self, output_dir: str = './output') -> Dict:
        import os
        os.makedirs(output_dir, exist_ok=True)

        global_shap = self.global_.shap_global_importance()
        global_perm = self.global_.permutation_importance()
        feature_stability = self.global_.feature_stability_analysis(global_shap)

        n_adv_samples = min(10, len(self.X))
        adversarial_result = self.adversarial.batch_detect(n_samples=n_adv_samples)

        report_path = os.path.join(output_dir, 'interpretability_report.html')
        report_result = self.reporter.export_html_report(report_path)

        return {
            'global_shap': global_shap,
            'global_permutation': global_perm,
            'feature_stability': feature_stability,
            'adversarial': adversarial_result,
            'report': report_result,
        }
