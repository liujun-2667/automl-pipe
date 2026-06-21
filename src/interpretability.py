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

            # 处理不同的 SHAP 返回格式
            # 旧版本: list of arrays [class0_shap, class1_shap]
            # 新版本 SHAP 0.52+: 3D array (n_samples, n_features, n_classes)
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # 新版本: (1, n_features, n_classes)，取正类的 shap values
                n_classes = shap_values.shape[2]
                if n_classes > 1:
                    shap_values = shap_values[0, :, 1]  # 取正类
                else:
                    shap_values = shap_values[0, :, 0]
            elif isinstance(shap_values, list):
                # 旧版本: 列表格式
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values
            elif shap_values.ndim > 1:
                shap_values = shap_values[0]

            shap_values = np.asarray(shap_values).flatten()

            base_value = None
            if hasattr(self._shap_explainer, 'expected_value'):
                base_value = self._shap_explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if len(base_value) > 1:
                        base_value = base_value[1]  # 取正类
                    else:
                        base_value = base_value[0]

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

    def explain_compare(self, sample_idx1: int, sample_idx2: int) -> Dict:
        """对比两个样本的解释结果，分析决策差异

        Args:
            sample_idx1: 第一个样本索引（通常为正例）
            sample_idx2: 第二个样本索引（通常为反例）

        Returns:
            包含两个样本的完整解释结果和决策差异分析
        """
        try:
            shap1 = self.explain_shap(sample_idx1)
            lime1 = self.explain_lime(sample_idx1)
            ice1 = self.explain_ice(sample_idx1)

            shap2 = self.explain_shap(sample_idx2)
            lime2 = self.explain_lime(sample_idx2)
            ice2 = self.explain_ice(sample_idx2)

            # 计算预测值
            X1 = self.X.iloc[[sample_idx1]]
            X2 = self.X.iloc[[sample_idx2]]
            pred1 = float(self.model.predict(X1)[0])
            pred2 = float(self.model.predict(X2)[0])
            has_proba = hasattr(self.model, 'predict_proba')
            if has_proba:
                pred1 = float(self.model.predict_proba(X1)[0][1])
                pred2 = float(self.model.predict_proba(X2)[0][1])

            # 计算一致性评分（即使某些explainer失败）
            consistency1 = 0.0
            consistency2 = 0.0
            if 'error' not in shap1 and 'error' not in lime1 and 'error' not in ice1:
                consistency1 = self.compute_consistency_score(shap1, lime1, ice1).get('mean_consistency', 0.0)
            if 'error' not in shap2 and 'error' not in lime2 and 'error' not in ice2:
                consistency2 = self.compute_consistency_score(shap2, lime2, ice2).get('mean_consistency', 0.0)

            # 决策差异分析（至少需要SHAP）
            decision_diff = None
            if 'error' not in shap1 and 'error' not in shap2:
                decision_diff = self._analyze_decision_difference(shap1, shap2, lime1, lime2)

            return {
                'sample1': {
                    'idx': sample_idx1,
                    'prediction': pred1,
                    'shap': shap1,
                    'lime': lime1,
                    'ice': ice1,
                    'consistency_score': consistency1,
                },
                'sample2': {
                    'idx': sample_idx2,
                    'prediction': pred2,
                    'shap': shap2,
                    'lime': lime2,
                    'ice': ice2,
                    'consistency_score': consistency2,
                },
                'decision_difference': decision_diff,
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_decision_difference(self, shap1: Dict, shap2: Dict,
                                     lime1: Dict, lime2: Dict) -> Dict:
        """分析两个样本间的决策差异，找出贡献方向相反的特征"""
        try:
            shap_contribs1 = {c['feature']: c['shap_value'] for c in shap1.get('feature_contributions', [])}
            shap_contribs2 = {c['feature']: c['shap_value'] for c in shap2.get('feature_contributions', [])}

            lime_weights1 = {}
            for w in lime1.get('feature_weights', []):
                feat = w['feature']
                for fn in self.feature_names:
                    if fn in feat:
                        lime_weights1[fn] = w['weight']
                        break

            lime_weights2 = {}
            for w in lime2.get('feature_weights', []):
                feat = w['feature']
                for fn in self.feature_names:
                    if fn in feat:
                        lime_weights2[fn] = w['weight']
                        break

            opposite_features = []
            all_features = list(set(list(shap_contribs1.keys()) + list(shap_contribs2.keys())))

            for feat in all_features:
                s1 = shap_contribs1.get(feat, 0)
                s2 = shap_contribs2.get(feat, 0)

                if (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0):
                    diff_abs = abs(s1 - s2)
                    opposite_features.append({
                        'feature': feat,
                        'sample1_shap': float(s1),
                        'sample2_shap': float(s2),
                        'sample1_lime': float(lime_weights1.get(feat, 0)),
                        'sample2_lime': float(lime_weights2.get(feat, 0)),
                        'diff_abs': float(diff_abs),
                        'direction': '正→负' if s1 > 0 else '负→正',
                    })

            opposite_features.sort(key=lambda x: x['diff_abs'], reverse=True)

            pred1 = shap1.get('prediction', 0)
            pred2 = shap2.get('prediction', 0)

            return {
                'prediction_diff': float(abs(pred1 - pred2)),
                'prediction_sample1': float(pred1),
                'prediction_sample2': float(pred2),
                'n_opposite_features': len(opposite_features),
                'opposite_features': opposite_features,
                'summary': f"两个样本预测值差异为 {abs(pred1 - pred2):.4f}，共发现 {len(opposite_features)} 个贡献方向相反的特征",
            }
        except Exception as e:
            return {'error': str(e)}


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

            # 处理不同的 SHAP 返回格式
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # 新版本 SHAP 0.52+: (n_samples, n_features, n_classes)
                n_classes = shap_values.shape[2]
                if n_classes > 1:
                    shap_values = shap_values[:, :, 1]  # 取正类
                else:
                    shap_values = shap_values[:, :, 0]
            elif isinstance(shap_values, list):
                # 旧版本: 列表格式
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
                'grid_values': pdp_result['grid_values'][0].tolist(),
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

            grid1 = pdp_result['grid_values'][0]
            grid2 = pdp_result['grid_values'][1]
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

    def compute_pdp_multi(self, feature: str, models: List[Tuple[str, Any]],
                          grid_resolution: int = 30) -> Dict:
        """同时计算多个模型的一维PDP，用于对比不同模型的偏依赖模式

        Args:
            feature: 要分析的特征名
            models: [(model_name, model_instance), ...] 模型列表
            grid_resolution: 网格分辨率

        Returns:
            包含各模型PDP结果的字典
        """
        try:
            if feature not in self.X.columns:
                return {'error': f'Feature {feature} not found'}

            if not pd.api.types.is_numeric_dtype(self.X[feature]):
                return {'error': f'Feature {feature} is not numeric'}

            if not models:
                return {'error': 'No models provided'}

            results = []
            errors = []
            for model_name, model in models:
                try:
                    pdp_result = partial_dependence(
                        model, self.X, features=[feature],
                        grid_resolution=grid_resolution,
                        method='brute',
                    )
                    pdp_vals = pdp_result['average'][0]
                    results.append({
                        'model_name': model_name,
                        'grid_values': pdp_result['grid_values'][0].tolist(),
                        'partial_dependence': pdp_vals.tolist(),
                        'pdp_values': pdp_vals,
                    })
                except Exception as e:
                    errors.append(f"{model_name}: {str(e)}")
                    continue

            if not results:
                return {'error': f'All models failed to compute PDP. Errors: {"; ".join(errors)}'}

            return {
                'feature': feature,
                'n_models': len(results),
                'model_results': results,
                'grid_values_common': results[0]['grid_values'] if results else [],
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

    def _perturb_single_feature(self, sample_idx: int, feature: str) -> pd.DataFrame:
        """仅对单个特征进行扰动，其他特征保持不变"""
        sample = self.X.iloc[[sample_idx]].copy()

        if feature not in sample.columns:
            return sample

        col_type = self.column_types.get(feature, '')
        series = self.X[feature]

        if pd.api.types.is_numeric_dtype(series) or col_type == 'numeric':
            feat_std = series.std()
            if feat_std > 0 and not np.isnan(feat_std):
                perturbation = self.rng.normal(0, 0.5 * feat_std)
                sample[feature] = sample[feature].values[0] + perturbation
        else:
            value_counts = series.value_counts()
            if len(value_counts) >= 2:
                current_val = sample[feature].values[0]
                other_vals = [v for v in value_counts.index if v != current_val]
                if other_vals:
                    sample[feature] = other_vals[0]

        return sample

    def trace_attribution_path(self, sample_idx: int, original_shap: Dict) -> Dict:
        """追踪是哪个特征的扰动导致了解释排序变化最大

        Args:
            sample_idx: 样本索引
            original_shap: 原始样本的SHAP解释结果

        Returns:
            包含归因不稳定特征的字典
        """
        try:
            if 'error' in original_shap or 'top_features' not in original_shap:
                return {'error': 'Invalid original SHAP result'}

            original_rank = original_shap['top_features']

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

            base_tau = 1.0

            # 性能优化：一次性生成所有特征的扰动样本，用同一个解释器批量计算
            perturbed_samples = []
            feat_indices = []

            for i, feat in enumerate(self.feature_names):
                try:
                    perturbed_sample = self._perturb_single_feature(sample_idx, feat)
                    perturbed_samples.append(perturbed_sample)
                    feat_indices.append(i)
                except Exception:
                    continue

            if not perturbed_samples:
                return {
                    'trigger_feature': None,
                    'unstable_feature': None,
                    'original_tau': base_tau,
                    'min_tau': base_tau,
                    'max_tau_drop': 0.0,
                    'feature_tau_drops': {},
                }

            # 将所有扰动样本拼接成一个DataFrame，一次性计算
            all_perturbed = pd.concat(perturbed_samples, ignore_index=True)

            # 使用已有的 local_interpreter 实例，避免重复创建
            # 注意：需要用 all_perturbed 创建一个临时解释器来计算这些样本的SHAP
            # 但是为了避免重复创建explainer，我们可以直接复用模型，重新计算
            try:
                import shap
                if self.local_interpreter._shap_explainer is not None:
                    # 复用已有explainer，直接计算
                    shap_values = self.local_interpreter._shap_explainer.shap_values(all_perturbed)

                    # 处理不同的SHAP返回格式
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        n_classes = shap_values.shape[2]
                        if n_classes > 1:
                            shap_values = shap_values[:, :, 1]
                        else:
                            shap_values = shap_values[:, :, 0]
                    elif isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                    shap_values = np.asarray(shap_values)
                    if shap_values.ndim == 1:
                        shap_values = shap_values.reshape(1, -1)
                else:
                    # SHAP不可用，直接返回空结果
                    return {
                        'trigger_feature': None,
                        'unstable_feature': None,
                        'original_tau': base_tau,
                        'min_tau': base_tau,
                        'max_tau_drop': 0.0,
                        'feature_tau_drops': {},
                    }
            except Exception:
                return {
                    'trigger_feature': None,
                    'unstable_feature': None,
                    'original_tau': base_tau,
                    'min_tau': base_tau,
                    'max_tau_drop': 0.0,
                    'feature_tau_drops': {},
                }

            tau_drops = []
            feat_drops_dict = {}

            for i, feat in enumerate(self.feature_names):
                if i >= len(shap_values):
                    continue

                try:
                    # 计算扰动后的特征重要性排序
                    abs_shap = np.abs(shap_values[i])
                    sorted_indices = np.argsort(-abs_shap)
                    perturbed_rank = [self.feature_names[j] for j in sorted_indices]

                    tau = top3_kendall(original_rank, perturbed_rank)
                    tau_drop = base_tau - tau

                    tau_drops.append({
                        'feature': feat,
                        'tau_after_perturb': float(tau),
                        'tau_drop': float(tau_drop),
                    })
                    feat_drops_dict[feat] = float(tau_drop)
                except Exception:
                    continue

            if not tau_drops:
                return {
                    'trigger_feature': None,
                    'unstable_feature': None,
                    'original_tau': base_tau,
                    'min_tau': base_tau,
                    'max_tau_drop': 0.0,
                    'feature_tau_drops': {},
                }

            tau_drops.sort(key=lambda x: x['tau_drop'], reverse=True)
            max_drop = tau_drops[0]
            min_tau = min(d['tau_after_perturb'] for d in tau_drops)

            return {
                'trigger_feature': max_drop['feature'],
                'unstable_feature': max_drop['feature'],
                'original_tau': base_tau,
                'min_tau': float(min_tau),
                'max_tau_drop': float(max_drop['tau_drop']),
                'feature_tau_drops': feat_drops_dict,
            }
        except Exception as e:
            return {'error': str(e)}

    def detect_sensitivity(self, sample_idx: int, enable_attribution_path: bool = True) -> Dict:
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

            attribution_path = None
            if enable_attribution_path:
                attribution_path = self.trace_attribution_path(sample_idx, original_shap)

            return self._compute_sensitivity_result(
                sample_idx,
                original_shap, original_lime, original_ice,
                perturbed_shap, perturbed_lime, perturbed_ice,
                perturbed_sample,
                attribution_path=attribution_path,
            )
        except Exception as e:
            return {'error': str(e)}

    def _compute_sensitivity_result(
        self, sample_idx: int,
        original_shap: Dict, original_lime: Dict, original_ice: Dict,
        perturbed_shap: Dict, perturbed_lime: Dict, perturbed_ice: Dict,
        perturbed_sample: pd.DataFrame,
        attribution_path: Optional[Dict] = None,
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

        trigger_feature = None
        max_tau_drop = 0.0
        if attribution_path and 'unstable_feature' in attribution_path:
            trigger_feature = attribution_path['unstable_feature']
            max_tau_drop = attribution_path.get('max_tau_drop', 0.0)

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
            'trigger_feature': trigger_feature,
            'max_tau_drop': float(max_tau_drop),
            'attribution_path': attribution_path,
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

    def batch_detect(self, sample_indices: Optional[List[int]] = None, n_samples: int = 10,
                     enable_attribution_path: bool = True, attribution_only_sensitive: bool = True) -> Dict:
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
        sensitive_indices = []

        # 第一轮：先检测所有样本的敏感性
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

                # 先不传 attribution_path，标记一下哪些是敏感的
                result = self._compute_sensitivity_result(
                    idx,
                    original_shap, original_lime, original_ice,
                    perturbed_shap, perturbed_lime, perturbed_ice,
                    perturbed_sample,
                    attribution_path=None,
                )
                results.append(result)

                if result.get('is_sensitive', False):
                    sensitive_indices.append((i, idx, original_shap))

            except Exception:
                continue

        # 第二轮：只对敏感样本做归因路径追踪（如果启用）
        if enable_attribution_path and sensitive_indices and attribution_only_sensitive:
            for i, idx, original_shap in sensitive_indices:
                try:
                    attribution_path = self.trace_attribution_path(idx, original_shap)
                    # 更新对应结果中的 trigger_feature
                    for r in results:
                        if r.get('sample_idx') == idx:
                            if 'error' not in attribution_path:
                                r['trigger_feature'] = attribution_path.get('trigger_feature')
                                r['max_tau_drop'] = attribution_path.get('max_tau_drop')
                            break
                except Exception:
                    continue
        elif enable_attribution_path and not attribution_only_sensitive:
            # 对所有样本做归因（原来的行为，保留兼容）
            for i, idx in enumerate(sample_indices):
                # 找到对应的 original_shap
                original_shap = None
                for r in results:
                    if r.get('sample_idx') == idx:
                        original_shap = r.get('_original_shap')
                        break

                if original_shap is None:
                    continue

                try:
                    attribution_path = self.trace_attribution_path(idx, original_shap)
                    for r in results:
                        if r.get('sample_idx') == idx:
                            if 'error' not in attribution_path:
                                r['trigger_feature'] = attribution_path.get('trigger_feature')
                                r['max_tau_drop'] = attribution_path.get('max_tau_drop')
                            break
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
                 column_types: Optional[Dict] = None, random_state: int = 42,
                 all_models: Optional[List[Tuple[str, Any]]] = None):
        self.model = model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.column_types = column_types or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        if all_models is None:
            self.all_models = [(model_name, model)]
        else:
            self.all_models = all_models

        self.local_interpreter = LocalInterpreter(model, X, task_type, self.feature_names, random_state)
        self.global_interpreter = GlobalInterpreter(model, X, y, task_type, self.feature_names, random_state)
        self.adversarial_explainer = AdversarialExplainer(
            model, X, task_type, self.feature_names, column_types, random_state
        )

    def compute_model_consistency(self, n_samples: int = 5) -> Dict:
        """计算多个模型间的解释一致性

        Args:
            n_samples: 用于计算一致性的随机样本数

        Returns:
            包含模型间一致性矩阵和热力图数据的字典
        """
        try:
            if len(self.all_models) < 2:
                return {'error': 'Need at least 2 models for consistency comparison'}

            n_samples = min(n_samples, len(self.X))
            sample_indices = self.rng.choice(len(self.X), size=n_samples, replace=False).tolist()

            model_rankings = {}

            for model_name, model in self.all_models:
                try:
                    local_interp = LocalInterpreter(
                        model, self.X, self.task_type, self.feature_names, self.random_state
                    )

                    feature_importances = np.zeros(len(self.feature_names))

                    for idx in sample_indices:
                        shap_res = local_interp.explain_shap(idx)
                        if 'error' in shap_res or 'feature_contributions' not in shap_res:
                            continue

                        for contrib in shap_res['feature_contributions']:
                            feat = contrib['feature']
                            if feat in self.feature_names:
                                feat_idx = self.feature_names.index(feat)
                                feature_importances[feat_idx] += abs(contrib['shap_value'])

                    avg_importance = feature_importances / max(len(sample_indices), 1)
                    sorted_indices = np.argsort(-avg_importance)
                    ranking = [self.feature_names[i] for i in sorted_indices]

                    model_rankings[model_name] = ranking
                except Exception:
                    continue

            if len(model_rankings) < 2:
                return {'error': 'Failed to compute rankings for enough models'}

            model_names = list(model_rankings.keys())
            n_models = len(model_names)

            consistency_matrix = np.zeros((n_models, n_models))
            low_consistency_pairs = []

            for i in range(n_models):
                for j in range(n_models):
                    if i == j:
                        consistency_matrix[i, j] = 1.0
                    elif i < j:
                        rank1 = model_rankings[model_names[i]]
                        rank2 = model_rankings[model_names[j]]

                        all_feats = list(set(rank1[:10] + rank2[:10]))
                        if len(all_feats) < 2:
                            tau = 1.0
                        else:
                            r1 = [rank1.index(f) if f in rank1 else len(rank1) for f in all_feats]
                            r2 = [rank2.index(f) if f in rank2 else len(rank2) for f in all_feats]
                            try:
                                tau, _ = kendalltau(r1, r2)
                                tau = float(tau) if not np.isnan(tau) else 0.0
                            except Exception:
                                tau = 0.0

                        consistency_matrix[i, j] = tau
                        consistency_matrix[j, i] = tau

                        if tau < 0.4:
                            low_consistency_pairs.append({
                                'model1': model_names[i],
                                'model2': model_names[j],
                                'kendall_tau': float(tau),
                                'is_divergent': True,
                            })

            # 转换为标准格式，找出最低一致性对
            avg_consistency = float(np.mean(consistency_matrix[np.triu_indices(n_models, k=1)])) if n_models > 1 else 0.0

            min_tau = 1.0
            lowest_pair = (None, None)
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    if consistency_matrix[i, j] < min_tau:
                        min_tau = consistency_matrix[i, j]
                        lowest_pair = (model_names[i], model_names[j])

            return {
                'model_names': model_names,
                'consistency_matrix': consistency_matrix,
                'low_consistency_pairs': low_consistency_pairs,
                'n_samples_used': n_samples,
                'has_divergence': len(low_consistency_pairs) > 0,
                'avg_consistency': avg_consistency,
                'lowest_pair': lowest_pair,
                'lowest_tau': float(min_tau),
            }
        except Exception as e:
            return {'error': str(e)}

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

        compare_results = []
        try:
            all_consistencies = []
            for idx in range(min(len(self.X), 20)):
                shap_res = self.local_interpreter.explain_shap(idx)
                lime_res = self.local_interpreter.explain_lime(idx)
                ice_res = self.local_interpreter.explain_ice(idx)
                if 'error' not in shap_res and 'error' not in lime_res and 'error' not in ice_res:
                    consistency = self.local_interpreter.compute_consistency_score(shap_res, lime_res, ice_res)
                    all_consistencies.append((idx, consistency['mean_consistency']))

            if len(all_consistencies) >= 4:
                all_consistencies.sort(key=lambda x: x[1])
                low_consistency_idx = all_consistencies[0][0]
                low_consistency_idx2 = all_consistencies[1][0]
                high_consistency_idx = all_consistencies[-1][0]
                high_consistency_idx2 = all_consistencies[-2][0]

                compare_high = self.local_interpreter.explain_compare(high_consistency_idx, high_consistency_idx2)
                compare_low = self.local_interpreter.explain_compare(low_consistency_idx, low_consistency_idx2)

                if 'error' not in compare_high:
                    compare_results.append(('高一致性样本对', compare_high))
                if 'error' not in compare_low:
                    compare_results.append(('低一致性样本对', compare_low))
        except Exception:
            pass

        pdp_multi_result = None
        try:
            if len(self.all_models) > 1 and pdp_feature:
                top_models_for_pdp = [(name, model) for name, model, _ in self.all_models[:3]] if isinstance(self.all_models[0], tuple) and len(self.all_models[0]) == 3 else [(name, model) for name, model in self.all_models[:3]]
                pdp_multi_result = self.global_interpreter.compute_pdp_multi(pdp_feature, top_models_for_pdp)
        except Exception:
            pass

        model_consistency = None
        try:
            if len(self.all_models) >= 2:
                model_consistency = self.compute_model_consistency(n_samples=5)
        except Exception:
            pass

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

        if pdp_multi_result and 'model_results' in pdp_multi_result:
            pdp_fig = go.Figure()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            for i, model_res in enumerate(pdp_multi_result['model_results']):
                color = colors[i % len(colors)]
                pdp_fig.add_trace(go.Scatter(
                    x=model_res['grid_values'],
                    y=model_res['partial_dependence'],
                    mode='lines+markers',
                    name=model_res['model_name'],
                    line=dict(color=color, width=3),
                ))
            pdp_fig.update_layout(
                title=f'PDP偏依赖图对比（TOP-{pdp_multi_result["n_models"]}模型）- {pdp_multi_result["feature"]}',
                xaxis_title=pdp_multi_result['feature'],
                yaxis_title='预测值（偏依赖）',
                height=400,
                legend_title='模型',
            )
            html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(pdp_fig)}</div>')
        elif 'partial_dependence' in pdp_result:
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

        if compare_results:
            html_parts.append('<h2>🔄 样本对比模式</h2>')
            for compare_label, compare_res in compare_results:
                try:
                    s1 = compare_res.get('sample1', {})
                    s2 = compare_res.get('sample2', {})
                    diff = compare_res.get('decision_difference')

                    idx1 = s1.get('idx', '?')
                    idx2 = s2.get('idx', '?')

                    html_parts.append(f'<h3>{compare_label}: 样本 #{idx1} vs 样本 #{idx2}</h3>')

                    pred1 = s1.get('prediction', 0)
                    pred2 = s2.get('prediction', 0)
                    pred_diff = abs(pred1 - pred2)
                    n_opposite = diff.get('n_opposite_features', 0) if diff else 0

                    html_parts.append(f'''
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">样本 #{idx1} 预测值</div>
        <div class="metric-value">{pred1:.4f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">样本 #{idx2} 预测值</div>
        <div class="metric-value">{pred2:.4f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">预测值差异</div>
        <div class="metric-value">{pred_diff:.4f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">贡献方向相反特征数</div>
        <div class="metric-value">{n_opposite}</div>
    </div>
</div>
''')

                    # SHAP瀑布图对比
                    html_parts.append('<h4>📊 SHAP瀑布图对比</h4>')
                    shap_available = False
                    for sample_key, sample_data in [('s1', s1), ('s2', s2)]:
                        shap_res = sample_data.get('shap', {})
                        if 'feature_contributions' in shap_res:
                            shap_available = True
                            contribs = shap_res['feature_contributions'][:10]
                            shap_waterfall = go.Figure()
                            cum_val = shap_res.get('base_value', 0)
                            shap_waterfall.add_trace(go.Waterfall(
                                orientation='v',
                                measure=['absolute'] + ['relative'] * len(contribs) + ['total'],
                                x=['E[f(x)]'] + [f"{c['feature']}={c['value']:.2f}" for c in contribs] + ['f(x)'],
                                y=[cum_val] + [c['shap_value'] for c in contribs] + [sum([cum_val] + [c['shap_value'] for c in contribs])],
                                connector={'line': {'color': 'rgb(100,100,100)'}},
                            ))
                            shap_waterfall.update_layout(
                                title=f'样本 #{sample_data.get("idx", "?")} SHAP瀑布图',
                                height=400,
                            )
                            html_parts.append(f'<div class="plot-container" style="display: inline-block; width: 48%; margin: 1%;">{self._generate_plotly_figure(shap_waterfall)}</div>')
                    if not shap_available:
                        html_parts.append('<div class="suggestion-box"><strong>💡 提示：</strong>SHAP解释器不可用，无法展示SHAP瀑布图。</div>')

                    # LIME特征权重对比
                    html_parts.append('<h4>📊 LIME特征权重对比</h4>')
                    lime_available = False
                    for sample_key, sample_data in [('s1', s1), ('s2', s2)]:
                        lime_res = sample_data.get('lime', {})
                        if 'feature_weights' in lime_res:
                            lime_available = True
                            weights = lime_res['feature_weights'][:10]
                            lime_fig = go.Figure()
                            lime_fig.add_trace(go.Bar(
                                x=[w['weight'] for w in reversed(weights)],
                                y=[w['feature'] for w in reversed(weights)],
                                orientation='h',
                                marker_color=['#e74c3c' if w['weight'] < 0 else '#27ae60' for w in reversed(weights)],
                            ))
                            lime_fig.update_layout(
                                title=f'样本 #{sample_data.get("idx", "?")} LIME特征权重',
                                xaxis_title='特征权重',
                                height=400,
                            )
                            html_parts.append(f'<div class="plot-container" style="display: inline-block; width: 48%; margin: 1%;">{self._generate_plotly_figure(lime_fig)}</div>')
                    if not lime_available:
                        html_parts.append('<div class="suggestion-box"><strong>💡 提示：</strong>LIME解释器不可用，无法展示LIME特征权重。</div>')

                    # 决策差异分析
                    if diff and diff.get('opposite_features'):
                        html_parts.append('<h4>⚡ 决策差异分析 - 贡献方向相反的特征</h4>')
                        html_parts.append('<table><tr><th>特征</th><th>样本1 SHAP</th><th>样本2 SHAP</th><th>差异绝对值</th><th>方向变化</th></tr>')
                        for feat in diff['opposite_features'][:10]:
                            color1 = '#27ae60' if feat.get('sample1_shap', 0) > 0 else '#e74c3c'
                            color2 = '#27ae60' if feat.get('sample2_shap', 0) > 0 else '#e74c3c'
                            html_parts.append(
                                f"<tr><td>{feat.get('feature', '?')}</td>"
                                f"<td style='color: {color1};'>{feat.get('sample1_shap', 0):.4f}</td>"
                                f"<td style='color: {color2};'>{feat.get('sample2_shap', 0):.4f}</td>"
                                f"<td><strong>{feat.get('diff_abs', 0):.4f}</strong></td>"
                                f"<td>{feat.get('direction', '-')}</td></tr>"
                            )
                        html_parts.append('</table>')
                    else:
                        if not diff:
                            html_parts.append('<div class="suggestion-box"><strong>💡 提示：</strong>SHAP解释器不可用，无法进行决策差异分析。</div>')
                        elif n_opposite == 0:
                            html_parts.append('<div class="suggestion-box"><strong>📌 说明：</strong>两个样本没有发现贡献方向相反的特征。</div>')
                except Exception as e:
                    html_parts.append(f'<div class="suggestion-box"><strong>⚠️ 样本对比渲染失败：</strong>{str(e)}</div>')

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
            html_parts.append('<h3>详细检测结果</h3><table><tr><th>样本</th><th>状态</th><th>Mean τ</th><th>SHAP τ</th><th>LIME τ</th><th>ICE τ</th><th>预测变化</th><th>触发特征</th></tr>')
            for r in adv['sample_results']:
                badge = '<span class="badge-sensitive">敏感</span>' if r['is_sensitive'] else '<span class="badge-ok">稳定</span>'
                trigger_feat = r.get('trigger_feature', '-')
                if trigger_feat is None:
                    trigger_feat = '-'
                html_parts.append(
                    f"<tr><td>#{r['sample_idx']}</td><td>{badge}</td>"
                    f"<td>{r['mean_kendall_tau']:.3f}</td>"
                    f"<td>{r['shap_tau']:.3f}</td>"
                    f"<td>{r['lime_tau']:.3f}</td>"
                    f"<td>{r['ice_tau']:.3f}</td>"
                    f"<td>{r['prediction_change']:.4f}</td>"
                    f"<td><strong>{trigger_feat}</strong></td></tr>"
                )
            html_parts.append('</table>')

        if model_consistency and 'model_names' in model_consistency:
            html_parts.append('<h2>🔗 模型间解释一致性对比</h2>')

            mc = model_consistency
            html_parts.append(f'''
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">参与对比模型数</div>
        <div class="metric-value">{len(mc["model_names"])}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">用于计算的样本数</div>
        <div class="metric-value">{mc["n_samples_used"]}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">是否存在解释分歧</div>
        <div class="metric-value">{'是' if mc["has_divergence"] else '否'}</div>
    </div>
</div>
''')

            heatmap_fig = go.Figure(data=go.Heatmap(
                z=mc['consistency_matrix'],
                x=mc['model_names'],
                y=mc['model_names'],
                text=[[f'{val:.3f}' for val in row] for row in mc['consistency_matrix']],
                texttemplate='%{text}',
                textfont={"size": 12},
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                hoverongaps=False,
            ))
            heatmap_fig.update_layout(
                title='模型间特征重要性排序一致性（Kendall τ）',
                xaxis_title='模型',
                yaxis_title='模型',
                height=500,
            )
            html_parts.append(f'<div class="plot-container">{self._generate_plotly_figure(heatmap_fig)}</div>')

            if mc.get('low_consistency_pairs'):
                html_parts.append('<h4>⚠️ 解释分歧模型对（τ < 0.4）</h4>')
                html_parts.append('<div class="suggestion-box">')
                html_parts.append('<strong>💡 提示：</strong>以下模型虽然性能接近，但决策逻辑可能有本质区别，建议谨慎选择。')
                html_parts.append('</div>')
                html_parts.append('<table><tr><th>模型1</th><th>模型2</th><th>Kendall τ</th><th>状态</th></tr>')
                for pair in mc['low_consistency_pairs']:
                    html_parts.append(
                        f"<tr><td>{pair['model1']}</td>"
                        f"<td>{pair['model2']}</td>"
                        f"<td style='color: #e74c3c;'><strong>{pair['kendall_tau']:.3f}</strong></td>"
                        f"<td><span class='badge-conflict'>解释分歧</span></td></tr>"
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
                 column_types: Optional[Dict] = None, random_state: int = 42,
                 all_models: Optional[List[Tuple[str, Any]]] = None):
        self.model = model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.column_types = column_types or {}
        self.random_state = random_state
        self.all_models = all_models

        self.local = LocalInterpreter(model, X, task_type, self.feature_names, random_state)
        self.global_ = GlobalInterpreter(model, X, y, task_type, self.feature_names, random_state)
        self.adversarial = AdversarialExplainer(
            model, X, task_type, self.feature_names, self.column_types, random_state
        )
        self.reporter = InterpretabilityReportExporter(
            model, X, y, task_type, model_name, self.feature_names,
            self.column_types, random_state, all_models=all_models
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
