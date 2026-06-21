"""
数据漂移检测与告警模块
- 单特征漂移检测: KS检验(数值型) + 卡方检验(分类型)
- 整体数据集漂移: PSI(Population Stability Index)
- 漂移判定与分级: Bonferroni校正 + PSI分级 + 综合告警
- 告警持久化: JSON文件存储历史告警
- 报告导出: HTML/PDF格式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde
import warnings
import json
import os
from datetime import datetime
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

EPSILON = 1e-10


class DriftDetector:
    """数据漂移检测器
    
    对参考数据集和新数据集进行多维度的漂移检测：
    - 数值型特征：KS检验计算统计量和p值
    - 分类型特征：卡方检验
    - 整体数据集：PSI指标
    
    Args:
        reference_data: 参考数据集DataFrame(通常是训练集/验证集)
        column_types: 列类型字典 {列名: 'numeric'/'categorical'/'date'/'text'/'id'}
        n_bins: PSI分桶数量，默认10(等频分桶)
        p_value_threshold: 单特征检验的原始p值阈值，默认0.05
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_types: Dict[str, str],
        n_bins: int = 10,
        p_value_threshold: float = 0.05,
    ):
        self.reference_data = reference_data.copy()
        self.column_types = column_types
        self.n_bins = n_bins
        self.p_value_threshold = p_value_threshold

        self._bin_edges: Dict[str, np.ndarray] = {}
        self._bin_edges_categorical: Dict[str, List[Any]] = {}
        self._reference_proportions: Dict[str, np.ndarray] = {}

        self._feature_columns = [
            col for col in reference_data.columns
            if col in column_types and column_types[col] in ('numeric', 'categorical')
        ]

        self._numeric_columns = [
            col for col in self._feature_columns
            if column_types[col] == 'numeric'
        ]
        self._categorical_columns = [
            col for col in self._feature_columns
            if column_types[col] == 'categorical'
        ]

        self._total_features = len(self._feature_columns)
        self._corrected_threshold = (
            p_value_threshold / self._total_features
            if self._total_features > 0
            else p_value_threshold
        )

        self._precompute_reference_bins()
        self._last_drift_result: Optional[Dict] = None

    def _precompute_reference_bins(self):
        """预计算参考数据集的分桶边界(用于PSI)"""
        for col in self._numeric_columns:
            series = self.reference_data[col].dropna()
            if len(series) == 0 or series.nunique() < 2:
                continue

            try:
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                edges = np.quantile(series.values, quantiles)
                edges = np.unique(edges)
                if len(edges) < 3:
                    edges = np.array([series.min() - 1e-9, series.max() + 1e-9])
                else:
                    edges[0] = edges[0] - 1e-9
                    edges[-1] = edges[-1] + 1e-9

                self._bin_edges[col] = edges

                counts, _ = np.histogram(series.values, bins=edges)
                props = counts / counts.sum()
                self._reference_proportions[col] = self._smooth_proportions(props)
            except Exception:
                continue

        for col in self._categorical_columns:
            series = self.reference_data[col].astype(str).dropna()
            if len(series) == 0:
                continue

            value_counts = series.value_counts()
            total = value_counts.sum()
            categories = value_counts.index.tolist()
            self._bin_edges_categorical[col] = categories

            props = np.array([value_counts.get(cat, 0) / total for cat in categories])
            self._reference_proportions[col] = self._smooth_proportions(props)

    @staticmethod
    def _smooth_proportions(props: np.ndarray) -> np.ndarray:
        """对占比做平滑处理，防止除零"""
        if np.any(props == 0):
            props = props + EPSILON
            props = props / props.sum()
        return props

    def _compute_psi_numeric(self, col: str, new_series: pd.Series) -> Dict:
        """计算单个数值型特征的PSI"""
        result = {'feature': col, 'type': 'numeric', 'psi_value': 0.0, 'bin_details': []}

        if col not in self._bin_edges or col not in self._reference_proportions:
            return result

        edges = self._bin_edges[col]
        ref_props = self._reference_proportions[col]

        new_clean = new_series.dropna()
        if len(new_clean) == 0:
            result['psi_value'] = float('inf')
            return result

        try:
            new_counts, _ = np.histogram(new_clean.values, bins=edges)
            new_total = new_counts.sum()

            if new_total == 0:
                result['psi_value'] = 0.0
                return result

            new_props = new_counts / new_total

            bin_details = []
            psi_total = 0.0
            for i in range(len(ref_props)):
                ref_p = ref_props[i]
                new_p = new_props[i] if i < len(new_props) else 0.0

                if new_p == 0:
                    bin_psi = 0.0
                else:
                    bin_psi = (new_p - ref_p) * np.log(new_p / ref_p)

                psi_total += bin_psi
                bin_details.append({
                    'bin_index': i,
                    'bin_left': float(edges[i]),
                    'bin_right': float(edges[i + 1]),
                    'reference_proportion': float(ref_p),
                    'new_proportion': float(new_p),
                    'psi_contribution': float(bin_psi),
                })

            result['psi_value'] = float(psi_total)
            result['bin_details'] = bin_details
            return result
        except Exception:
            return result

    def _compute_psi_categorical(self, col: str, new_series: pd.Series) -> Dict:
        """计算单个分类型特征的PSI"""
        result = {'feature': col, 'type': 'categorical', 'psi_value': 0.0, 'bin_details': []}

        if col not in self._bin_edges_categorical or col not in self._reference_proportions:
            return result

        categories = self._bin_edges_categorical[col]
        ref_props = self._reference_proportions[col]

        new_clean = new_series.astype(str).dropna()
        if len(new_clean) == 0:
            result['psi_value'] = float('inf')
            return result

        try:
            new_value_counts = new_clean.value_counts()
            new_total = len(new_clean)

            if new_total == 0:
                result['psi_value'] = 0.0
                return result

            bin_details = []
            psi_total = 0.0
            for i, cat in enumerate(categories):
                ref_p = ref_props[i]
                new_p = new_value_counts.get(cat, 0) / new_total

                if new_p == 0:
                    bin_psi = 0.0
                else:
                    bin_psi = (new_p - ref_p) * np.log(new_p / ref_p)

                psi_total += bin_psi
                bin_details.append({
                    'category': cat,
                    'reference_proportion': float(ref_p),
                    'new_proportion': float(new_p),
                    'psi_contribution': float(bin_psi),
                })

            unseen_categories = [c for c in new_value_counts.index if c not in categories]
            if unseen_categories:
                unseen_total = sum(new_value_counts[c] for c in unseen_categories)
                unseen_p = unseen_total / new_total
                ref_p_for_unseen = EPSILON
                bin_psi = (unseen_p - ref_p_for_unseen) * np.log(unseen_p / ref_p_for_unseen)
                psi_total += bin_psi
                bin_details.append({
                    'category': '__UNSEEN__',
                    'reference_proportion': float(ref_p_for_unseen),
                    'new_proportion': float(unseen_p),
                    'psi_contribution': float(bin_psi),
                    'unseen_categories': unseen_categories,
                })

            result['psi_value'] = float(psi_total)
            result['bin_details'] = bin_details
            return result
        except Exception:
            return result

    def _ks_test(self, col: str, new_series: pd.Series) -> Dict:
        """对数值型特征进行KS检验"""
        ref_clean = self.reference_data[col].dropna().values
        new_clean = new_series.dropna().values

        result = {
            'feature': col,
            'test': 'ks_2samp',
            'statistic': 0.0,
            'p_value': 1.0,
            'is_drifted': False,
            'direction': 'none',
        }

        if len(ref_clean) == 0 or len(new_clean) == 0:
            return result

        try:
            stat, p_val = ks_2samp(ref_clean, new_clean)
            result['statistic'] = float(stat)
            result['p_value'] = float(p_val)
            result['is_drifted'] = p_val < self._corrected_threshold

            ref_mean = float(np.mean(ref_clean))
            new_mean = float(np.mean(new_clean))
            if abs(ref_mean - new_mean) > 1e-9:
                if new_mean > ref_mean:
                    result['direction'] = 'mean_increase'
                else:
                    result['direction'] = 'mean_decrease'
            result['reference_mean'] = ref_mean
            result['new_mean'] = new_mean
            return result
        except Exception:
            return result

    def _chi_square_test(self, col: str, new_series: pd.Series) -> Dict:
        """对分类型特征进行卡方检验"""
        ref_clean = self.reference_data[col].astype(str).dropna()
        new_clean = new_series.astype(str).dropna()

        result = {
            'feature': col,
            'test': 'chi2',
            'statistic': 0.0,
            'p_value': 1.0,
            'is_drifted': False,
            'direction': 'none',
            'category_changes': [],
        }

        if len(ref_clean) == 0 or len(new_clean) == 0:
            return result

        try:
            all_categories = sorted(list(set(ref_clean.unique()) | set(new_clean.unique())))

            ref_counts = np.array([(ref_clean == c).sum() for c in all_categories], dtype=float)
            new_counts = np.array([(new_clean == c).sum() for c in all_categories], dtype=float)

            contingency_table = np.array([ref_counts, new_counts])
            if np.any(contingency_table.sum(axis=0) == 0):
                valid_cols = contingency_table.sum(axis=0) > 0
                contingency_table = contingency_table[:, valid_cols]
                all_categories = [c for i, c in enumerate(all_categories) if valid_cols[i]]

            if contingency_table.shape[1] < 2:
                return result

            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            result['statistic'] = float(chi2)
            result['p_value'] = float(p_val)
            result['is_drifted'] = p_val < self._corrected_threshold

            ref_total = ref_counts.sum()
            new_total = new_counts.sum()
            category_changes = []
            max_change_cat = None
            max_change_val = 0

            for i, cat in enumerate(all_categories):
                ref_prop = ref_counts[i] / ref_total if ref_total > 0 else 0
                new_prop = new_counts[i] / new_total if new_total > 0 else 0
                change = new_prop - ref_prop
                category_changes.append({
                    'category': cat,
                    'reference_proportion': float(ref_prop),
                    'new_proportion': float(new_prop),
                    'change': float(change),
                })
                if abs(change) > max_change_val:
                    max_change_val = abs(change)
                    max_change_cat = cat
                    result['direction'] = (
                        f"category_{'increase' if change > 0 else 'decrease'}"
                    )

            result['category_changes'] = category_changes
            result['most_changed_category'] = max_change_cat
            return result
        except Exception:
            return result

    @staticmethod
    def _psi_grade(psi_value: float) -> str:
        """PSI分级"""
        if psi_value < 0.1:
            return 'stable'
        elif psi_value < 0.25:
            return 'mild_drift'
        else:
            return 'severe_drift'

    def _grade_level(self, grade: str) -> int:
        """告警级别映射为数值"""
        levels = {'stable': 0, 'mild_drift': 1, 'severe_drift': 2}
        return levels.get(grade, 0)

    def _combine_alerts(
        self,
        psi_grade: str,
        drifted_feature_count: int,
        total_features: int,
    ) -> str:
        """综合PSI和单特征检测结果，确定最终告警级别"""
        psi_level = self._grade_level(psi_grade)
        final_level = psi_level

        drift_ratio = drifted_feature_count / max(total_features, 1)

        if drifted_feature_count >= 3 and psi_level == 0:
            final_level = max(final_level, 1)

        if drifted_feature_count >= 5 and psi_level <= 1:
            final_level = max(final_level, 1)

        if drift_ratio >= 0.3 and psi_level <= 1:
            final_level = max(final_level, 2)

        level_to_grade = {0: 'stable', 1: 'mild_drift', 2: 'severe_drift'}
        return level_to_grade.get(final_level, 'stable')

    def _generate_retraining_advice(
        self,
        overall_grade: str,
        psi_value: float,
        drifted_feature_count: int,
        total_features: int,
    ) -> Dict:
        """生成重训建议"""
        advice = {
            'action': 'continue_monitoring',
            'reason': '',
            'urgency': 'low',
        }

        if total_features == 0:
            advice['reason'] = '无可分析特征'
            return advice

        drift_ratio = drifted_feature_count / total_features

        if overall_grade == 'stable':
            advice['action'] = 'continue_monitoring'
            advice['urgency'] = 'low'
            advice['reason'] = '数据分布稳定，未见显著漂移'
        elif overall_grade == 'mild_drift':
            if psi_value < 0.2 and drift_ratio < 0.3:
                advice['action'] = 'monitor_closely'
                advice['urgency'] = 'medium'
                advice['reason'] = (
                    f'检测到轻度漂移: {drifted_feature_count}/{total_features} 个特征漂移 '
                    f'({drift_ratio:.0%}), PSI={psi_value:.4f}。建议密切监控，暂无需立即重训'
                )
            elif drift_ratio >= 0.3:
                advice['action'] = 'retrain_immediately'
                advice['urgency'] = 'high'
                advice['reason'] = (
                    f'漂移特征占比过高: {drifted_feature_count}/{total_features} 个特征漂移 '
                    f'({drift_ratio:.0%}, 超过30%阈值), PSI={psi_value:.4f}。建议立即重新训练模型!'
                )
            else:
                advice['action'] = 'consider_retrain'
                advice['urgency'] = 'medium'
                advice['reason'] = (
                    f'PSI接近阈值: {drifted_feature_count}/{total_features} 个特征漂移 ({drift_ratio:.0%}), '
                    f'PSI={psi_value:.4f}。建议准备重训'
                )
        else:
            advice['action'] = 'retrain_immediately'
            advice['urgency'] = 'high'
            advice['reason'] = (
                f'检测到严重漂移: {drifted_feature_count}/{total_features} 个特征漂移 '
                f'({drift_ratio:.0%}), PSI={psi_value:.4f}。建议立即重新训练模型!'
            )

        return advice

    def detect(self, new_data: pd.DataFrame) -> Dict:
        """对新数据集执行完整的漂移检测

        Args:
            new_data: 待检测的新数据集DataFrame

        Returns:
            完整的检测结果字典，包含:
            - timestamp: 检测时间戳
            - overall_psi: 整体PSI值(各特征PSI的均值)
            - overall_psi_grade: PSI分级
            - corrected_p_threshold: Bonferroni校正后的p值阈值
            - feature_tests: 各特征检验结果详情
            - feature_psi: 各特征PSI详情
            - drifted_features: 漂移特征列表
            - n_drifted: 漂移特征数量
            - n_total_features: 总特征数
            - overall_alert_level: 综合告警级别
            - alert_banner: 告警横幅文本
            - retraining_advice: 重训建议
            - distribution_data: 分布对比数据(用于绘图)
        """
        timestamp = datetime.now().isoformat()

        feature_tests = {}
        feature_psi = {}
        distribution_data = {}

        common_features = [
            col for col in self._feature_columns if col in new_data.columns
        ]

        for col in common_features:
            col_type = self.column_types[col]
            if col_type == 'numeric':
                feature_tests[col] = self._ks_test(col, new_data[col])
                feature_psi[col] = self._compute_psi_numeric(col, new_data[col])
            elif col_type == 'categorical':
                feature_tests[col] = self._chi_square_test(col, new_data[col])
                feature_psi[col] = self._compute_psi_categorical(col, new_data[col])

            distribution_data[col] = self._get_distribution_data(col, new_data[col])

        psi_values = [
            v['psi_value'] for v in feature_psi.values()
            if np.isfinite(v['psi_value'])
        ]
        overall_psi = self._compute_overall_psi(psi_values) if psi_values else 0.0
        overall_psi_grade = self._psi_grade(overall_psi)

        drifted_features = []
        drift_details = []
        for col, test in feature_tests.items():
            if test.get('is_drifted', False):
                drifted_features.append(col)
                direction = test.get('direction', 'none')
                drift_details.append({
                    'feature': col,
                    'test': test.get('test', ''),
                    'p_value': test.get('p_value', 1.0),
                    'statistic': test.get('statistic', 0.0),
                    'direction': direction,
                    'direction_desc': self._describe_direction(col, test),
                })

        n_drifted = len(drifted_features)
        n_total = len(common_features)

        overall_alert_level = self._combine_alerts(
            overall_psi_grade, n_drifted, n_total
        )

        alert_banner = self._build_alert_banner(
            overall_alert_level, overall_psi, overall_psi_grade,
            drifted_features, n_total, drift_details,
        )

        retraining_advice = self._generate_retraining_advice(
            overall_alert_level, overall_psi, n_drifted, n_total,
        )

        result = {
            'timestamp': timestamp,
            'overall_psi': overall_psi,
            'overall_psi_grade': overall_psi_grade,
            'corrected_p_threshold': self._corrected_threshold,
            'original_p_threshold': self.p_value_threshold,
            'feature_tests': feature_tests,
            'feature_psi': feature_psi,
            'drifted_features': drifted_features,
            'drift_details': drift_details,
            'n_drifted': n_drifted,
            'n_total_features': n_total,
            'common_features': common_features,
            'overall_alert_level': overall_alert_level,
            'alert_banner': alert_banner,
            'retraining_advice': retraining_advice,
            'distribution_data': distribution_data,
        }

        self._last_drift_result = result

        return result

    @staticmethod
    def _compute_overall_psi(psi_values: List[float]) -> float:
        """计算整体PSI，避免单个严重漂移被平均稀释

        计算逻辑:
        1. 简单均值 (mean_psi) - 反映整体漂移程度
        2. 最大值加权 (max_psi * 0.5) - 反映局部最严重漂移
        3. 第75百分位数 (p75_psi) - 反映多数特征的漂移水平
        取三者中的最大值作为整体PSI，确保局部严重漂移能被反映

        Args:
            psi_values: 各特征PSI值列表

        Returns:
            综合整体PSI值
        """
        if not psi_values:
            return 0.0

        psi_arr = np.array(psi_values)
        mean_psi = float(np.mean(psi_arr))
        max_psi = float(np.max(psi_arr))
        p75_psi = float(np.percentile(psi_arr, 75))

        overall = float(max(mean_psi, max_psi * 0.5, p75_psi))
        return overall

    def _get_distribution_data(self, col: str, new_series: pd.Series) -> Dict:
        """获取分布对比数据(用于绘图)"""
        col_type = self.column_types[col]
        result = {
            'feature': col,
            'type': col_type,
            'reference': {},
            'new': {},
        }

        ref_series = self.reference_data[col]
        new_clean = new_series.dropna()
        ref_clean = ref_series.dropna()

        if col_type == 'numeric':
            result['reference']['hist'], result['reference']['edges'] = np.histogram(
                ref_clean.values, bins=20, density=True
            )
            if len(new_clean) > 0:
                result['new']['hist'], result['new']['edges'] = np.histogram(
                    new_clean.values, bins=result['reference']['edges'], density=True
                )
            else:
                result['new']['hist'] = np.zeros_like(result['reference']['hist'])
                result['new']['edges'] = result['reference']['edges']
            result['reference']['mean'] = float(np.mean(ref_clean)) if len(ref_clean) else 0.0
            result['reference']['std'] = float(np.std(ref_clean)) if len(ref_clean) else 0.0
            result['new']['mean'] = float(np.mean(new_clean)) if len(new_clean) else 0.0
            result['new']['std'] = float(np.std(new_clean)) if len(new_clean) else 0.0
        elif col_type == 'categorical':
            ref_counts = ref_clean.astype(str).value_counts(normalize=True)
            new_counts = new_clean.astype(str).value_counts(normalize=True) if len(new_clean) else pd.Series(dtype=float)
            all_cats = sorted(list(set(ref_counts.index) | set(new_counts.index)))[:30]
            result['reference']['categories'] = all_cats
            result['reference']['proportions'] = [float(ref_counts.get(c, 0)) for c in all_cats]
            result['new']['categories'] = all_cats
            result['new']['proportions'] = [float(new_counts.get(c, 0)) for c in all_cats]

        return result

    def get_feature_drilldown(self, feature: str, new_data: pd.DataFrame) -> Dict:
        """漂移根因下钻：返回单个特征在参考集与新数据集的详细对比

        Args:
            feature: 特征名
            new_data: 待检测的新数据集

        Returns:
            包含描述统计、分位数对比、KDE/瀑布图数据的字典
        """
        if feature not in self.column_types:
            return {'error': f'特征 {feature} 不在列类型配置中'}

        if feature not in self.reference_data.columns:
            return {'error': f'参考数据中无特征 {feature}'}

        if feature not in new_data.columns:
            return {'error': f'新数据中无特征 {feature}'}

        col_type = self.column_types[feature]
        ref_series = self.reference_data[feature]
        new_series = new_data[feature]

        result = {'feature': feature, 'type': col_type}

        if col_type == 'numeric':
            result['stats'] = self._numeric_describe(ref_series, new_series)
            result['quantiles'] = self._quantile_comparison(ref_series, new_series)
            result['kde'] = self._kde_overlay_data(ref_series, new_series)
        elif col_type == 'categorical':
            result['stats'] = self._categorical_describe(ref_series, new_series)
            result['waterfall'] = self._category_waterfall(ref_series, new_series)

        return result

    @staticmethod
    def _numeric_describe(ref_series: pd.Series, new_series: pd.Series) -> Dict:
        """数值型特征描述统计对比"""
        def describe(s: pd.Series) -> Dict:
            clean = pd.to_numeric(s, errors='coerce').dropna()
            total = len(s)
            if len(clean) == 0:
                return {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0,
                    'missing_rate': 1.0 if total > 0 else 1.0,
                    'count': 0,
                }
            return {
                'mean': float(clean.mean()),
                'median': float(clean.median()),
                'std': float(clean.std()),
                'min': float(clean.min()),
                'max': float(clean.max()),
                'missing_rate': float(s.isna().mean()) if total > 0 else 1.0,
                'count': int(len(clean)),
            }
        return {'reference': describe(ref_series), 'new': describe(new_series)}

    @staticmethod
    def _quantile_comparison(ref_series: pd.Series, new_series: pd.Series) -> List[Dict]:
        """分位数对比 (5%/25%/50%/75%/95%)"""
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        ref_clean = pd.to_numeric(ref_series, errors='coerce').dropna()
        new_clean = pd.to_numeric(new_series, errors='coerce').dropna()
        rows = []
        for q in quantiles:
            ref_q = float(ref_clean.quantile(q)) if len(ref_clean) else 0.0
            new_q = float(new_clean.quantile(q)) if len(new_clean) else 0.0
            rows.append({
                'quantile': f'{int(q * 100)}%',
                'reference': ref_q,
                'new': new_q,
                'diff': float(new_q - ref_q),
            })
        return rows

    @staticmethod
    def _kde_overlay_data(ref_series: pd.Series, new_series: pd.Series) -> Dict:
        """KDE密度曲线叠加数据"""
        ref_clean = pd.to_numeric(ref_series, errors='coerce').dropna().values
        new_clean = pd.to_numeric(new_series, errors='coerce').dropna().values

        if len(ref_clean) < 2 or len(new_clean) < 2:
            return {'available': False, 'reason': '样本数不足'}

        try:
            lo = float(min(ref_clean.min(), new_clean.min()))
            hi = float(max(ref_clean.max(), new_clean.max()))
            if hi <= lo:
                return {'available': False, 'reason': '数据范围为空'}

            span = hi - lo
            grid = np.linspace(lo - span * 0.05, hi + span * 0.05, 100)

            ref_kde = gaussian_kde(ref_clean)
            new_kde = gaussian_kde(new_clean)

            return {
                'available': True,
                'grid': grid.tolist(),
                'ref_density': ref_kde(grid).tolist(),
                'new_density': new_kde(grid).tolist(),
            }
        except Exception as e:
            return {'available': False, 'reason': str(e)}

    @staticmethod
    def _categorical_describe(ref_series: pd.Series, new_series: pd.Series) -> Dict:
        """分类型特征描述统计对比"""
        def describe(s: pd.Series) -> Dict:
            clean = s.astype(str).dropna()
            vc = clean.value_counts()
            return {
                'n_unique': int(clean.nunique()),
                'top': str(vc.index[0]) if len(vc) else '',
                'top_freq': int(vc.iloc[0]) if len(vc) else 0,
                'missing_rate': float(s.isna().mean()) if len(s) else 1.0,
                'count': int(len(clean)),
            }
        return {'reference': describe(ref_series), 'new': describe(new_series)}

    @staticmethod
    def _category_waterfall(ref_series: pd.Series, new_series: pd.Series) -> List[Dict]:
        """分类占比变化瀑布图数据 (参考占比 -> 新占比的增减)"""
        ref_clean = ref_series.astype(str).dropna()
        new_clean = new_series.astype(str).dropna()
        ref_vc = ref_clean.value_counts(normalize=True)
        new_vc = new_clean.value_counts(normalize=True)
        all_cats = sorted(set(ref_vc.index) | set(new_vc.index))

        rows = []
        for cat in all_cats:
            ref_p = float(ref_vc.get(cat, 0.0))
            new_p = float(new_vc.get(cat, 0.0))
            rows.append({
                'category': cat,
                'reference': ref_p,
                'new': new_p,
                'delta': float(new_p - ref_p),
            })
        rows.sort(key=lambda x: abs(x['delta']), reverse=True)
        return rows

    def _describe_direction(self, col: str, test: Dict) -> str:
        """将方向枚举转换为可读描述"""
        direction = test.get('direction', 'none')
        col_type = self.column_types.get(col, '')

        if col_type == 'numeric':
            if direction == 'mean_increase':
                return (
                    f"均值上升 (参考: {test.get('reference_mean', 0):.4f} → "
                    f"新数据: {test.get('new_mean', 0):.4f})"
                )
            elif direction == 'mean_decrease':
                return (
                    f"均值下降 (参考: {test.get('reference_mean', 0):.4f} → "
                    f"新数据: {test.get('new_mean', 0):.4f})"
                )
        elif col_type == 'categorical':
            changes = test.get('category_changes', [])
            if changes:
                top_changes = sorted(changes, key=lambda x: abs(x['change']), reverse=True)[:3]
                descs = []
                for c in top_changes:
                    if abs(c['change']) < 0.01:
                        continue
                    if c['change'] > 0:
                        descs.append(
                            f"类别'{c['category']}'占比上升 "
                            f"({c['reference_proportion']:.1%} → {c['new_proportion']:.1%})"
                        )
                    else:
                        descs.append(
                            f"类别'{c['category']}'占比下降 "
                            f"({c['reference_proportion']:.1%} → {c['new_proportion']:.1%})"
                        )
                if descs:
                    return '; '.join(descs)

        return '分布偏移'

    def _build_alert_banner(
        self,
        alert_level: str,
        psi_value: float,
        psi_grade: str,
        drifted_features: List[str],
        n_total: int,
        drift_details: List[Dict],
    ) -> Dict:
        """构建告警横幅信息"""
        grade_cn = {
            'stable': '✅ 稳定',
            'mild_drift': '⚠️ 轻度漂移',
            'severe_drift': '🚨 严重漂移',
        }

        banner = {
            'level': alert_level,
            'label': grade_cn.get(alert_level, alert_level),
            'summary': '',
            'details': [],
            'drifted_feature_details': [],
        }

        if alert_level == 'stable':
            banner['summary'] = (
                f'数据分布稳定。整体PSI={psi_value:.4f} (级别: {self._psi_cn(psi_grade)}), '
                f'共检测 {n_total} 个特征，{len(drifted_features)} 个存在统计显著漂移'
            )
        else:
            banner['summary'] = (
                f'检测到{grade_cn.get(alert_level, "")}! '
                f'整体PSI={psi_value:.4f} (级别: {self._psi_cn(psi_grade)}), '
                f'{len(drifted_features)}/{n_total} 个特征存在统计显著漂移'
            )

            for d in drift_details[:10]:
                banner['drifted_feature_details'].append({
                    'feature': d['feature'],
                    'p_value': f"{d['p_value']:.2e}",
                    'direction_desc': d['direction_desc'],
                })

            if len(drifted_features) > 10:
                banner['details'].append(
                    f'另有 {len(drifted_features) - 10} 个特征存在漂移，详见下方分析'
                )

        return banner

    @staticmethod
    def _psi_cn(grade: str) -> str:
        mapping = {
            'stable': '稳定',
            'mild_drift': '轻度漂移',
            'severe_drift': '严重漂移',
        }
        return mapping.get(grade, grade)


class AlertStorage:
    """告警持久化存储 - JSON文件格式"""

    def __init__(self, storage_path: str = './drift_alerts.json'):
        self.storage_path = storage_path
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)) or '.', exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump({'alerts': []}, f, ensure_ascii=False, indent=2)

    def save_alert(self, detection_result: Dict, dataset_name: str = 'unknown') -> Dict:
        """保存一次检测告警记录"""
        summary = {
            'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
            'timestamp': detection_result.get('timestamp', datetime.now().isoformat()),
            'dataset_name': dataset_name,
            'overall_alert_level': detection_result.get('overall_alert_level', 'stable'),
            'overall_psi': detection_result.get('overall_psi', 0.0),
            'overall_psi_grade': detection_result.get('overall_psi_grade', 'stable'),
            'n_drifted': detection_result.get('n_drifted', 0),
            'n_total_features': detection_result.get('n_total_features', 0),
            'drifted_features': detection_result.get('drifted_features', []),
            'alert_summary': detection_result.get('alert_banner', {}).get('summary', ''),
            'retraining_action': detection_result.get('retraining_advice', {}).get('action', ''),
            'retraining_urgency': detection_result.get('retraining_advice', {}).get('urgency', ''),
            'retraining_reason': detection_result.get('retraining_advice', {}).get('reason', ''),
        }

        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['alerts'].append(summary)

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return summary

    def get_all_alerts(self, limit: Optional[int] = None) -> List[Dict]:
        """获取所有历史告警记录"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            alerts = data.get('alerts', [])
            alerts_sorted = sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
            if limit:
                return alerts_sorted[:limit]
            return alerts_sorted
        except Exception:
            return []

    def get_alerts_by_level(self, level: str, limit: Optional[int] = None) -> List[Dict]:
        """按告警级别筛选记录"""
        alerts = self.get_all_alerts()
        filtered = [a for a in alerts if a.get('overall_alert_level') == level]
        if limit:
            return filtered[:limit]
        return filtered

    def clear_alerts(self) -> int:
        """清空所有告警记录，返回删除的数量"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data.get('alerts', []))
            data['alerts'] = []
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return count
        except Exception:
            return 0


class DriftReportExporter:
    """漂移检测报告导出器 - HTML格式"""

    GRADE_COLORS = {
        'stable': '#27ae60',
        'mild_drift': '#f39c12',
        'severe_drift': '#e74c3c',
    }

    GRADE_LABELS = {
        'stable': '稳定',
        'mild_drift': '轻度漂移',
        'severe_drift': '严重漂移',
    }

    def __init__(self, detection_result: Dict):
        self.result = detection_result

    def export_html(self, output_path: Optional[str] = None) -> str:
        """导出HTML报告，返回HTML内容"""
        html = self._build_html()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        return html

    def _build_html(self) -> str:
        r = self.result
        alert_level = r.get('overall_alert_level', 'stable')
        color = self.GRADE_COLORS.get(alert_level, '#95a5a6')
        label = self.GRADE_LABELS.get(alert_level, alert_level)

        feature_rows = self._build_feature_rows()
        psi_rows = self._build_psi_rows()
        drift_rows = self._build_drift_feature_rows()

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>数据漂移检测报告</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid {color}; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  h3 {{ color: #7f8c8d; }}
  .banner {{ background: {color}; color: white; padding: 20px; border-radius: 10px;
             font-size: 18px; font-weight: bold; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .metric-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  .metric-value {{ font-size: 28px; font-weight: bold; color: {color}; }}
  .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #ecf0f1; }}
  tr:hover {{ background: #f1f8ff; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  .tag-stable {{ background: #d5f5e3; color: #186a3b; }}
  .tag-mild {{ background: #fdebd0; color: #9c640c; }}
  .tag-severe {{ background: #fadbd8; color: #922b21; }}
  .advice-box {{ background: white; border-left: 5px solid {color}; padding: 20px; border-radius: 5px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 20px 0; }}
  .timestamp {{ color: #95a5a6; font-size: 13px; }}
  .bar-bg {{ background: #ecf0f1; height: 8px; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ background: {color}; height: 100%; }}
</style>
</head>
<body>

<h1>📊 数据漂移检测报告</h1>
<p class="timestamp">生成时间: {r.get('timestamp', '')}</p>

<div class="banner">
  综合告警级别: {label} | {r.get('alert_banner', {}).get('summary', '')}
</div>

<h2>📈 整体指标概览</h2>
<div class="metrics">
  <div class="metric-card">
    <div class="metric-value">{r.get('overall_psi', 0):.4f}</div>
    <div class="metric-label">整体PSI (Population Stability Index)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{self.GRADE_LABELS.get(r.get('overall_psi_grade', ''), '')}</div>
    <div class="metric-label">PSI分级</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{r.get('n_drifted', 0)}/{r.get('n_total_features', 0)}</div>
    <div class="metric-label">漂移特征数 / 总特征数</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{r.get('corrected_p_threshold', 0):.4e}</div>
    <div class="metric-label">Bonferroni校正阈值</div>
  </div>
</div>

<div class="advice-box">
  <h3>💡 重训建议</h3>
  <p><strong>动作:</strong> {r.get('retraining_advice', {}).get('action', '')}
     &nbsp;&nbsp;|&nbsp;&nbsp; <strong>紧急度:</strong> {r.get('retraining_advice', {}).get('urgency', '')}</p>
  <p><strong>说明:</strong> {r.get('retraining_advice', {}).get('reason', '')}</p>
</div>

<h2>🔥 漂移特征详情</h2>
{drift_rows if drift_rows else '<p style="color: #27ae60;">✅ 未检测到显著漂移的特征</p>'}
{drift_rows and f'''
<table>
  <tr><th>特征名</th><th>检验方法</th><th>统计量</th><th>p值</th><th>漂移方向</th></tr>
  {drift_rows}
</table>
''' if drift_rows else ''}

<h2>📋 单特征检验结果汇总</h2>
<table>
  <tr><th>特征名</th><th>类型</th><th>检验方法</th><th>统计量</th><th>p值</th><th>是否漂移</th></tr>
  {feature_rows}
</table>

<h2>📊 PSI特征排名 (按漂移程度从高到低)</h2>
<table>
  <tr><th>特征名</th><th>PSI值</th><th>分级</th><th>可视化</th></tr>
  {psi_rows}
</table>

<h2>📝 附录: 术语说明</h2>
<ul>
  <li><strong>PSI</strong>: Population Stability Index，衡量总体分布稳定性的指标。&lt;0.1稳定，0.1-0.25轻度漂移，&gt;0.25严重漂移</li>
  <li><strong>KS检验</strong>: Kolmogorov-Smirnov检验，用于比较两个数值型分布是否存在显著差异</li>
  <li><strong>卡方检验</strong>: 用于比较两个分类型分布是否存在显著差异</li>
  <li><strong>Bonferroni校正</strong>: 多重检验时将p值阈值除以检验次数，控制假阳性率</li>
</ul>

</body>
</html>"""
        return html

    def _build_feature_rows(self) -> str:
        rows = []
        feature_tests = self.result.get('feature_tests', {})
        type_cn = {'numeric': '数值型', 'categorical': '分类型'}

        sorted_items = sorted(
            feature_tests.items(),
            key=lambda x: (x[1].get('p_value', 1.0), -x[1].get('statistic', 0.0))
        )

        for feat, test in sorted_items:
            ftype = type_cn.get('numeric' if test.get('test') == 'ks_2samp' else 'categorical', '未知')
            is_drifted = test.get('is_drifted', False)
            tag_class = 'tag-severe' if is_drifted else 'tag-stable'
            tag_text = '漂移' if is_drifted else '稳定'

            rows.append(
                f"<tr>"
                f"<td><strong>{feat}</strong></td>"
                f"<td>{ftype}</td>"
                f"<td>{test.get('test', '')}</td>"
                f"<td>{test.get('statistic', 0):.4f}</td>"
                f"<td>{test.get('p_value', 1):.4e}</td>"
                f"<td><span class='tag {tag_class}'>{tag_text}</span></td>"
                f"</tr>"
            )
        return '\n'.join(rows)

    def _build_psi_rows(self) -> str:
        rows = []
        feature_psi = self.result.get('feature_psi', {})

        valid_items = [
            (f, v) for f, v in feature_psi.items()
            if np.isfinite(v.get('psi_value', 0))
        ]
        sorted_items = sorted(valid_items, key=lambda x: x[1]['psi_value'], reverse=True)

        max_psi = max((v['psi_value'] for _, v in sorted_items), default=1.0)

        for feat, psi in sorted_items:
            val = psi['psi_value']
            grade = DriftDetector._psi_grade(val)
            grade_label = self.GRADE_LABELS.get(grade, grade)
            if grade == 'stable':
                tag_class = 'tag-stable'
            elif grade == 'mild_drift':
                tag_class = 'tag-mild'
            else:
                tag_class = 'tag-severe'

            bar_width = min(100, (val / max(max_psi, 0.25)) * 100)
            bar_color = self.GRADE_COLORS.get(grade, '#95a5a6')

            rows.append(
                f"<tr>"
                f"<td><strong>{feat}</strong></td>"
                f"<td>{val:.4f}</td>"
                f"<td><span class='tag {tag_class}'>{grade_label}</span></td>"
                f"<td><div class='bar-bg'><div class='bar-fill' style='width:{bar_width:.0f}%; background:{bar_color};'></div></div></td>"
                f"</tr>"
            )
        return '\n'.join(rows)

    def _build_drift_feature_rows(self) -> str:
        drift_details = self.result.get('drift_details', [])
        if not drift_details:
            return ''

        rows = []
        for d in drift_details:
            rows.append(
                f"<tr>"
                f"<td><strong>{d['feature']}</strong></td>"
                f"<td>{d['test']}</td>"
                f"<td>{d['statistic']:.4f}</td>"
                f"<td>{d['p_value']:.4e}</td>"
                f"<td>{d['direction_desc']}</td>"
                f"</tr>"
            )
        return '\n'.join(rows)


def compute_weighted_psi(feature_psi: Dict, feature_weights: Dict) -> Optional[float]:
    """计算特征重要性加权PSI

    每个特征的PSI值乘以其归一化后的重要性权重再求和。
    若无有效权重（如尚未完成特征选择），返回 None。

    Args:
        feature_psi: 各特征PSI详情 {feature: {'psi_value': float}}
        feature_weights: 特征重要性权重 {feature: weight}

    Returns:
        加权PSI值；若无有效权重则返回 None
    """
    if not feature_weights:
        return None

    relevant = {}
    for feat, w in feature_weights.items():
        if feat not in feature_psi:
            continue
        psi_info = feature_psi[feat]
        psi_val = psi_info.get('psi_value', 0.0) if isinstance(psi_info, dict) else psi_info
        if not np.isfinite(psi_val):
            continue
        try:
            w_f = float(w)
        except (TypeError, ValueError):
            continue
        if w_f < 0 or not np.isfinite(w_f):
            continue
        relevant[feat] = w_f

    if not relevant:
        return None

    total_w = sum(relevant.values())
    if total_w <= 0:
        return None

    weighted = 0.0
    for feat, w in relevant.items():
        psi_info = feature_psi[feat]
        psi_val = psi_info.get('psi_value', 0.0) if isinstance(psi_info, dict) else psi_info
        weighted += float(psi_val) * (w / total_w)
    return float(weighted)


class SlidingWindowDriftMonitor:
    """滑动窗口漂移监控器

    维护一个时间滑动窗口，每次窗口滑动时自动执行漂移检测并记录结果。
    窗口内数据不足窗口大小时用已有数据填充，并在结果中标注
    "数据不足，结果仅供参考"。
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_types: Dict[str, str],
        window_size: int = 1000,
        step_size: int = 100,
        n_bins: int = 10,
        p_value_threshold: float = 0.05,
    ):
        self.reference_data = reference_data.copy()
        self.column_types = column_types
        self.window_size = max(1, int(window_size))
        self.step_size = max(1, int(step_size))
        self.n_bins = n_bins
        self.p_value_threshold = p_value_threshold

        self._buffer_parts: List[pd.DataFrame] = []
        self._window_start = 0
        self._total_seen = 0
        self._detector = DriftDetector(
            reference_data, column_types, n_bins, p_value_threshold
        )

    @property
    def feature_columns(self) -> List[str]:
        return self._detector._feature_columns

    def add_data(self, new_data: pd.DataFrame) -> int:
        """追加新数据到缓冲区，返回累计见到的总行数"""
        if new_data is None or len(new_data) == 0:
            return self._total_seen
        self._buffer_parts.append(new_data.copy())
        self._total_seen += len(new_data)
        return self._total_seen

    def _get_full_buffer(self) -> pd.DataFrame:
        if not self._buffer_parts:
            return pd.DataFrame()
        return pd.concat(self._buffer_parts, ignore_index=True)

    def get_current_window(self) -> pd.DataFrame:
        """获取当前窗口内的数据"""
        full = self._get_full_buffer()
        if full.empty:
            return pd.DataFrame()
        end = min(self._window_start + self.window_size, len(full))
        return full.iloc[self._window_start:end].copy()

    def detect_current_window(self) -> Dict:
        """对当前窗口执行漂移检测"""
        window = self.get_current_window()
        data_insufficient = len(window) < self.window_size

        if window.empty:
            return {
                'error': '窗口内暂无数据',
                'data_insufficient': True,
                'window_note': '数据不足，结果仅供参考',
            }

        feature_cols = [c for c in self.feature_columns if c in window.columns]
        window_filtered = window[feature_cols].copy()

        if window_filtered.empty:
            return {
                'error': '窗口内无可分析特征',
                'data_insufficient': True,
                'window_note': '数据不足，结果仅供参考',
            }

        result = self._detector.detect(window_filtered)
        result['data_insufficient'] = data_insufficient
        result['window_size'] = self.window_size
        result['actual_window_size'] = len(window)
        result['window_start'] = self._window_start
        result['window_end'] = self._window_start + len(window)
        result['total_seen'] = self._total_seen
        result['detection_mode'] = 'sliding_window'
        if data_insufficient:
            result['window_note'] = '数据不足，结果仅供参考'
        return result

    def slide(self) -> bool:
        """窗口向前滑动一个步长，返回是否成功滑动"""
        full = self._get_full_buffer()
        next_start = self._window_start + self.step_size
        if next_start >= len(full):
            return False
        self._window_start = next_start
        return True

    def can_slide(self) -> bool:
        full = self._get_full_buffer()
        return (self._window_start + self.step_size) < len(full)

    def reset(self):
        self._buffer_parts = []
        self._window_start = 0
        self._total_seen = 0

    def get_status(self) -> Dict:
        full = self._get_full_buffer()
        return {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'window_start': self._window_start,
            'buffer_size': len(full),
            'total_seen': self._total_seen,
            'can_slide': self.can_slide(),
            'remaining_in_window': max(0, len(full) - self._window_start),
        }


class DriftTrendTracker:
    """漂移趋势追踪器

    按时间顺序存储每次检测的PSI值和漂移特征数量，并支持识别
    连续超过警戒线的区间。
    """

    STABLE_THRESHOLD = 0.1
    WARNING_THRESHOLD = 0.25

    def __init__(self):
        self._records: List[Dict] = []

    def record(self, detection_result: Dict,
               weighted_psi: Optional[float] = None) -> Dict:
        """记录一次检测结果"""
        if weighted_psi is not None and isinstance(weighted_psi, float) \
                and np.isnan(weighted_psi):
            weighted_psi = None

        rec = {
            'seq': len(self._records) + 1,
            'timestamp': detection_result.get(
                'timestamp', datetime.now().isoformat()
            ),
            'overall_psi': float(detection_result.get('overall_psi', 0.0)),
            'weighted_psi': (
                float(weighted_psi) if weighted_psi is not None else None
            ),
            'n_drifted': int(detection_result.get('n_drifted', 0)),
            'n_total_features': int(detection_result.get('n_total_features', 0)),
            'overall_alert_level': detection_result.get(
                'overall_alert_level', 'stable'
            ),
            'data_insufficient': detection_result.get('data_insufficient', False),
        }
        self._records.append(rec)
        return rec

    def get_records(self) -> List[Dict]:
        return list(self._records)

    def clear(self):
        self._records = []

    def __len__(self):
        return len(self._records)

    def get_warning_streaks(
        self,
        warning_threshold: Optional[float] = None,
        min_consecutive: int = 3,
    ) -> List[Dict]:
        """找出连续超过警戒线的检测区间"""
        threshold = self.WARNING_THRESHOLD if warning_threshold is None else warning_threshold
        streaks = []
        start = None
        count = 0

        for i, rec in enumerate(self._records):
            over = rec['overall_psi'] > threshold
            if over:
                if start is None:
                    start = i
                    count = 1
                else:
                    count += 1
            else:
                if start is not None and count >= min_consecutive:
                    streaks.append(self._make_streak(start, i - 1, count))
                start = None
                count = 0

        if start is not None and count >= min_consecutive:
            streaks.append(self._make_streak(start, len(self._records) - 1, count))
        return streaks

    def _make_streak(self, start_idx: int, end_idx: int, length: int) -> Dict:
        return {
            'start_seq': self._records[start_idx]['seq'],
            'end_seq': self._records[end_idx]['seq'],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': length,
        }

    def get_max_streak(self, warning_threshold: Optional[float] = None) -> int:
        streaks = self.get_warning_streaks(warning_threshold, min_consecutive=1)
        return max((s['length'] for s in streaks), default=0)
