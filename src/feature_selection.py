"""
特征重要性评估与选择模块
- 随机森林特征重要性(Gini)
- 排列重要性(Permutation)
- L1正则化特征重要性
- 三种方法交集筛选
- 可视化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """特征重要性分析器 - 三种方法评估"""

    def __init__(
        self,
        task_type: str = 'binary',
        n_estimators: int = 100,
        random_state: int = 42,
        cv: int = 5,
    ):
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cv = cv

        self.rf_importance_ = None
        self.perm_importance_ = None
        self.l1_importance_ = None
        self.feature_names_ = None

    def _get_model_for_rf(self):
        if self.task_type in ['binary', 'multiclass']:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=15,
                min_samples_leaf=5,
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=15,
                min_samples_leaf=5,
            )

    def _get_model_for_l1(self):
        if self.task_type in ['binary', 'multiclass']:
            return LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            return Lasso(
                alpha=1.0,
                max_iter=5000,
                random_state=self.random_state,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureImportanceAnalyzer':
        """计算三种特征重要性"""
        self.feature_names_ = X.columns.tolist()
        self.rf_importance_ = self._random_forest_importance(X, y)
        self.perm_importance_ = self._permutation_importance(X, y)
        self.l1_importance_ = self._l1_importance(X, y)
        return self

    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """方法一: 随机森林Gini重要性"""
        model = self._get_model_for_rf()
        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=self.feature_names_,
            name='rf_importance'
        ).sort_values(ascending=False)

        return importance

    def _permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """方法二: 排列重要性"""
        model = self._get_model_for_rf()
        model.fit(X, y)

        if self.task_type == 'binary':
            scoring = 'roc_auc'
        elif self.task_type == 'multiclass':
            scoring = 'f1_macro'
        else:
            scoring = 'neg_mean_squared_error'

        result = permutation_importance(
            model, X, y,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1,
            scoring=scoring
        )

        importance = pd.Series(
            result.importances_mean,
            index=self.feature_names_,
            name='perm_importance'
        )
        importance = importance - importance.min()
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance.sort_values(ascending=False)

    def _l1_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """方法三: L1正则化系数绝对值"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = self._get_model_for_l1()
        model.fit(X_scaled, y)

        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                coef = np.mean(np.abs(model.coef_), axis=0)
            else:
                coef = np.abs(model.coef_)
        else:
            coef = np.zeros(len(self.feature_names_))

        importance = pd.Series(
            coef,
            index=self.feature_names_,
            name='l1_importance'
        )
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return importance.sort_values(ascending=False)

    def get_all_importances(self) -> pd.DataFrame:
        """获取三种方法的重要性DataFrame"""
        df = pd.DataFrame({
            'random_forest': self.rf_importance_,
            'permutation': self.perm_importance_,
            'l1_regularization': self.l1_importance_,
        })
        df = df.reindex(self.rf_importance_.index)
        return df

    def get_ranked_features(self, method: str = 'random_forest') -> pd.Series:
        """获取指定方法的特征排名"""
        if method == 'random_forest':
            return self.rf_importance_
        elif method == 'permutation':
            return self.perm_importance_
        elif method == 'l1_regularization':
            return self.l1_importance_
        else:
            raise ValueError(f"未知方法: {method}")

    def get_top_k_features(self, method: str, k: int) -> List[str]:
        """获取Top-K特征"""
        ranked = self.get_ranked_features(method)
        return ranked.head(k).index.tolist()


class IntersectionFeatureSelector:
    """交集特征选择器 - 至少被两种方法认为重要的特征"""

    def __init__(
        self,
        n_methods_required: int = 2,
        auto_threshold: float = 0.8,
    ):
        self.n_methods_required = n_methods_required
        self.auto_threshold = auto_threshold

        self.selected_features_ = []
        self.method_scores_ = None

    def fit(
        self,
        analyzer: FeatureImportanceAnalyzer,
        n_features: Optional[int] = None,
        auto: bool = True,
    ) -> 'IntersectionFeatureSelector':
        """
        选择特征

        Args:
            analyzer: 特征重要性分析器
            n_features: 手动指定保留特征数，None则自动选择
            auto: 是否自动选择累计重要性达阈值的最少特征数
        """
        all_features = analyzer.feature_names_

        top_counts = {feat: 0 for feat in all_features}

        for method in ['random_forest', 'permutation', 'l1_regularization']:
            ranked = analyzer.get_ranked_features(method)
            if auto:
                k = self._get_auto_k(ranked, self.auto_threshold)
            elif n_features is not None:
                k = min(n_features, len(all_features))
            else:
                k = len(all_features) // 2

            top_features = ranked.head(k).index.tolist()
            for feat in top_features:
                top_counts[feat] += 1

        selected = [feat for feat, count in top_counts.items() if count >= self.n_methods_required]

        if len(selected) < 5 and len(all_features) >= 5:
            selected = sorted(top_counts.items(), key=lambda x: -x[1])[:max(5, len(all_features) // 4)]
            selected = [feat for feat, _ in selected]

        if len(selected) == 0:
            selected = all_features[:10] if len(all_features) >= 10 else all_features

        avg_importance = {}
        for feat in selected:
            scores = []
            for method in ['random_forest', 'permutation', 'l1_regularization']:
                ranked = analyzer.get_ranked_features(method)
                if feat in ranked.index:
                    scores.append(ranked[feat])
            avg_importance[feat] = np.mean(scores) if scores else 0

        selected_sorted = sorted(selected, key=lambda x: -avg_importance[x])

        self.selected_features_ = selected_sorted
        self.method_scores_ = top_counts
        return self

    def _get_auto_k(self, ranked: pd.Series, threshold: float) -> int:
        """自动选择累计重要性达阈值的最少特征数"""
        total = ranked.sum()
        if total == 0:
            return len(ranked) // 2

        cumulative = (ranked / total).cumsum()
        k = (cumulative <= threshold).sum() + 1
        return max(min(k, len(ranked)), 5)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征后的DataFrame"""
        return X[self.selected_features_].copy()

    def fit_transform(self, analyzer: FeatureImportanceAnalyzer, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(analyzer, **kwargs)
        return self.transform(X)

    def get_venn_data(self, analyzer: FeatureImportanceAnalyzer, n_features: Optional[int] = None) -> Dict:
        """获取Venn图数据"""
        if n_features is None:
            n_features = min(len(self.selected_features_) + 10, len(analyzer.feature_names_))

        rf_set = set(analyzer.get_top_k_features('random_forest', n_features))
        perm_set = set(analyzer.get_top_k_features('permutation', n_features))
        l1_set = set(analyzer.get_top_k_features('l1_regularization', n_features))

        rf_only = rf_set - perm_set - l1_set
        perm_only = perm_set - rf_set - l1_set
        l1_only = l1_set - rf_set - perm_set

        rf_perm = (rf_set & perm_set) - l1_set
        rf_l1 = (rf_set & l1_set) - perm_set
        perm_l1 = (perm_set & l1_set) - rf_set

        all_three = rf_set & perm_set & l1_set

        return {
            'rf_only': sorted(list(rf_only)),
            'perm_only': sorted(list(perm_only)),
            'l1_only': sorted(list(l1_only)),
            'rf_perm': sorted(list(rf_perm)),
            'rf_l1': sorted(list(rf_l1)),
            'perm_l1': sorted(list(perm_l1)),
            'all_three': sorted(list(all_three)),
        }

    def get_selected_count(self) -> int:
        """获取选中特征数"""
        return len(self.selected_features_)
