"""
自动特征工程模块
- 数值型特征变换
- 分类型特征变换
- 日期型特征变换
- 文本型特征变换
- 特征过滤（方差过滤、高相关过滤）
- 可序列化Pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
import copy

warnings.filterwarnings('ignore')


class LogTransformer(BaseEstimator, TransformerMixin):
    """对数变换器"""

    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
        else:
            self.columns = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = np.log1p(X)
            result.columns = [f"{col}_log" for col in X.columns]
            return result
        return np.log1p(X)


class PolynomialCrossTransformer(BaseEstimator, TransformerMixin):
    """多项式交叉特征生成器（两两相乘）"""

    def __init__(self):
        self.columns = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            self.feature_names = []
            for i in range(len(self.columns)):
                for j in range(i + 1, len(self.columns)):
                    self.feature_names.append(f"{self.columns[i]}_x_{self.columns[j]}")
        else:
            self.columns = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        n_features = X_arr.shape[1]
        cross_features = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                cross_features.append(X_arr[:, i] * X_arr[:, j])

        if len(cross_features) == 0:
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(index=X.index)
            return np.array([]).reshape(X_arr.shape[0], 0)

        result = np.column_stack(cross_features)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=self.feature_names, index=X.index)
        return result


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """缺失值指示器"""

    def __init__(self):
        self.columns = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            self.feature_names = [f"{col}_is_missing" for col in self.columns]
        else:
            self.columns = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X.isna().astype(int)
            result.columns = self.feature_names
            return result
        return pd.isna(X).astype(int)


class BinningTransformer(BaseEstimator, TransformerMixin):
    """等频分箱变换器"""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.binner = None
        self.columns = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            self.feature_names = [f"{col}_bin_{self.n_bins}" for col in self.columns]
        else:
            self.columns = None

        self.binner = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy='quantile',
            subsample=None
        )
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.binner.fit(X_arr)
        return self

    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        result = self.binner.transform(X_arr)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=self.feature_names, index=X.index)
        return result


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """频次编码器"""

    def __init__(self):
        self.freq_maps = {}
        self.columns = None

    def fit(self, X, y=None):
        self.freq_maps = {}
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            for col in X.columns:
                freq = X[col].value_counts(normalize=True).to_dict()
                self.freq_maps[col] = freq
        else:
            for i in range(X.shape[1]):
                unique, counts = np.unique(X[:, i], return_counts=True)
                freq = dict(zip(unique, counts / len(X)))
                self.freq_maps[i] = freq
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            for col in X.columns:
                result[col] = X[col].map(self.freq_maps.get(col, {})).fillna(0)
            result.columns = [f"{col}_freq" for col in X.columns]
            return result
        else:
            result = np.zeros_like(X, dtype=float)
            for i in range(X.shape[1]):
                for j, val in enumerate(X[:, i]):
                    result[j, i] = self.freq_maps.get(i, {}).get(val, 0)
            return result


class TargetEncoder(BaseEstimator, TransformerMixin):
    """目标编码器（带贝叶斯平滑）"""

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None
        self.columns = None

    def fit(self, X, y):
        self.encodings = {}
        if isinstance(y, pd.Series):
            y = y.values

        y_numeric = y.astype(float) if not np.issubdtype(y.dtype, np.number) else y
        self.global_mean = np.mean(y_numeric)

        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            for col in X.columns:
                categories = X[col].values
                cat_stats = {}
                for cat in np.unique(categories):
                    mask = categories == cat
                    n = mask.sum()
                    mean = np.mean(y_numeric[mask])
                    smoothed = (n * mean + self.smoothing * self.global_mean) / (n + self.smoothing)
                    cat_stats[cat] = smoothed
                self.encodings[col] = cat_stats
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            for col in X.columns:
                result[col] = X[col].map(self.encodings.get(col, {})).fillna(self.global_mean)
            result.columns = [f"{col}_target" for col in X.columns]
            return result
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """日期特征提取器"""

    def __init__(self):
        self.columns = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            self.feature_names = []
            for col in self.columns:
                self.feature_names.extend([
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_is_weekend",
                    f"{col}_days_since_today",
                    f"{col}_quarter",
                ])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            features = []
            feature_names = []
            today = pd.Timestamp.today().normalize()
            for col in X.columns:
                dt_series = pd.to_datetime(X[col], errors='coerce')
                col_features = pd.DataFrame({
                    f"{col}_year": dt_series.dt.year,
                    f"{col}_month": dt_series.dt.month,
                    f"{col}_day": dt_series.dt.day,
                    f"{col}_dayofweek": dt_series.dt.dayofweek,
                    f"{col}_is_weekend": (dt_series.dt.dayofweek >= 5).astype(int),
                    f"{col}_days_since_today": (today - dt_series.dt.normalize()).dt.days,
                    f"{col}_quarter": dt_series.dt.quarter,
                })
                features.append(col_features)
            if features:
                return pd.concat(features, axis=1)
            return pd.DataFrame(index=X.index)
        return X


class TfidfTextTransformer(BaseEstimator, TransformerMixin):
    """TF-IDF文本向量化器"""

    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.vectorizers = {}
        self.columns = None
        self.feature_names = []

    def fit(self, X, y=None):
        self.vectorizers = {}
        self.feature_names = []
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            for col in X.columns:
                texts = X[col].fillna('').astype(str).tolist()
                vec = TfidfVectorizer(max_features=self.max_features, stop_words=None)
                vec.fit(texts)
                self.vectorizers[col] = vec
                for feat in vec.get_feature_names_out():
                    self.feature_names.append(f"{col}_tfidf_{feat}")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            all_features = []
            for col in X.columns:
                if col in self.vectorizers:
                    texts = X[col].fillna('').astype(str).tolist()
                    tfidf_matrix = self.vectorizers[col].transform(texts)
                    feature_names = [f"{col}_tfidf_{f}" for f in self.vectorizers[col].get_feature_names_out()]
                    df_tfidf = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=feature_names,
                        index=X.index
                    )
                    all_features.append(df_tfidf)
            if all_features:
                return pd.concat(all_features, axis=1)
            return pd.DataFrame(index=X.index)
        return X


class VarianceFilter(BaseEstimator, TransformerMixin):
    """方差过滤（删除方差为零的特征）"""

    def __init__(self):
        self.keep_cols = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            variances = X.var()
            self.keep_cols = variances[variances > 0].index.tolist()
            self.feature_names = self.keep_cols
        else:
            variances = np.var(X, axis=0)
            self.keep_cols = np.where(variances > 0)[0]
            self.feature_names = []
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.keep_cols].copy()
        return X[:, self.keep_cols]


class HighCorrelationFilter(BaseEstimator, TransformerMixin):
    """高相关过滤（删除相关系数绝对值大于阈值的特征对中的一个）"""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.keep_cols = None
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
            self.keep_cols = [col for col in X.columns if col not in to_drop]
            self.feature_names = self.keep_cols
        else:
            corr_matrix = np.abs(np.corrcoef(X.T))
            n_features = X.shape[1]
            to_drop = set()
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if corr_matrix[i, j] > self.threshold:
                        to_drop.add(j)
            self.keep_cols = [i for i in range(n_features) if i not in to_drop]
            self.feature_names = []
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.keep_cols].copy()
        return X[:, self.keep_cols]


class AutoFeatureEngineer:
    """自动特征工程器"""

    def __init__(
        self,
        column_types: Dict[str, str],
        task_type: str = 'binary',
        text_strategy: str = 'tfidf',
        enable_poly_cross: bool = True,
        corr_threshold: float = 0.95,
        max_tfidf_features: int = 100,
        n_bins: int = 5,
        target_encoder_smoothing: float = 10.0,
    ):
        self.column_types = column_types
        self.task_type = task_type
        self.text_strategy = text_strategy
        self.enable_poly_cross = enable_poly_cross
        self.corr_threshold = corr_threshold
        self.max_tfidf_features = max_tfidf_features
        self.n_bins = n_bins
        self.target_encoder_smoothing = target_encoder_smoothing

        self.pipeline = None
        self.feature_names_ = []
        self.transformers_info = []

        self._is_fitted = False
        self._numeric_imputer = None
        self._scaler = None
        self._binner = None
        self._log_transformer = None
        self._log_cols = []
        self._missing_indicator = None
        self._missing_cols = []
        self._poly_cross = None
        self._cat_imputer = None
        self._ohe = None
        self._low_card_cats = []
        self._ohe_feature_names = []
        self._target_encoder = None
        self._high_card_cats = []
        self._freq_encoder = None
        self._date_extractor = None
        self._tfidf = None
        self._var_filter = None
        self._corr_filter = None
        self._numeric_cols = []
        self._cat_cols = []
        self._date_cols = []
        self._text_cols = []

    def _get_numeric_columns(self) -> List[str]:
        return [c for c, t in self.column_types.items() if t == 'numeric']

    def _get_categorical_columns(self) -> List[str]:
        return [c for c, t in self.column_types.items() if t == 'categorical']

    def _get_date_columns(self) -> List[str]:
        return [c for c, t in self.column_types.items() if t == 'date']

    def _get_text_columns(self) -> List[str]:
        return [c for c, t in self.column_types.items() if t == 'text']

    def _get_id_columns(self) -> List[str]:
        return [c for c, t in self.column_types.items() if t == 'id']

    def _analyze_skew(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, bool]:
        """分析数值列偏度，判断是否需要对数变换"""
        log_cols = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                log_cols[col] = False
                continue
            all_positive = (series > 0).all()
            skewness = stats.skew(series)
            log_cols[col] = all_positive and skewness > 2
        return log_cols

    def _get_low_cardinality_cats(self, df: pd.DataFrame, cat_cols: List[str]) -> List[str]:
        """获取低基数分类列（<=10类别）"""
        return [c for c in cat_cols if df[c].nunique() <= 10]

    def _get_high_cardinality_cats(self, df: pd.DataFrame, cat_cols: List[str]) -> List[str]:
        """获取高基数分类列（>10类别）"""
        return [c for c in cat_cols if df[c].nunique() > 10]

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self._numeric_cols = self._get_numeric_columns()
        self._cat_cols = self._get_categorical_columns()
        self._date_cols = self._get_date_columns()
        self._text_cols = self._get_text_columns()

        self.transformers_info = []
        all_transformed = []

        if self._numeric_cols:
            numeric_df = df[self._numeric_cols].copy()

            self._numeric_imputer = SimpleImputer(strategy='median')
            imputed_numeric = pd.DataFrame(
                self._numeric_imputer.fit_transform(numeric_df),
                columns=self._numeric_cols,
                index=df.index
            )
            self.transformers_info.append(('numeric_imputer', 'median'))

            self._scaler = StandardScaler()
            scaled = pd.DataFrame(
                self._scaler.fit_transform(imputed_numeric),
                columns=[f"{c}_scaled" for c in self._numeric_cols],
                index=df.index
            )
            all_transformed.append(scaled)
            self.transformers_info.append(('standard_scaler', self._numeric_cols))

            self._binner = BinningTransformer(n_bins=self.n_bins)
            binned = self._binner.fit_transform(imputed_numeric)
            all_transformed.append(binned)
            self.transformers_info.append(('binning', f'{self.n_bins} bins'))

            log_cols_dict = self._analyze_skew(df, self._numeric_cols)
            self._log_cols = [c for c, need in log_cols_dict.items() if need]
            if self._log_cols:
                self._log_transformer = LogTransformer()
                log_df = self._log_transformer.fit_transform(imputed_numeric[self._log_cols])
                all_transformed.append(log_df)
                self.transformers_info.append(('log_transform', self._log_cols))
            else:
                self._log_transformer = None

            self._missing_cols = [c for c in self._numeric_cols if df[c].isna().any()]
            if self._missing_cols:
                self._missing_indicator = MissingIndicatorTransformer()
                missing_df = self._missing_indicator.fit_transform(numeric_df[self._missing_cols])
                all_transformed.append(missing_df)
                self.transformers_info.append(('missing_indicator', self._missing_cols))
            else:
                self._missing_indicator = None

            if self.enable_poly_cross and len(self._numeric_cols) < 20 and len(self._numeric_cols) >= 2:
                self._poly_cross = PolynomialCrossTransformer()
                poly_df = self._poly_cross.fit_transform(imputed_numeric)
                all_transformed.append(poly_df)
                self.transformers_info.append(('polynomial_cross', f'{len(self._numeric_cols)} features'))
            else:
                self._poly_cross = None

        if self._cat_cols:
            cat_df = df[self._cat_cols].copy()
            self._cat_imputer = SimpleImputer(strategy='most_frequent')
            imputed_cat = pd.DataFrame(
                self._cat_imputer.fit_transform(cat_df),
                columns=self._cat_cols,
                index=df.index
            )
            self.transformers_info.append(('categorical_imputer', 'most_frequent'))

            self._low_card_cats = self._get_low_cardinality_cats(df, self._cat_cols)
            if self._low_card_cats:
                from sklearn.preprocessing import OneHotEncoder
                self._ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                ohe_result = self._ohe.fit_transform(imputed_cat[self._low_card_cats])
                self._ohe_feature_names = self._ohe.get_feature_names_out(self._low_card_cats).tolist()
                ohe_df = pd.DataFrame(ohe_result, columns=self._ohe_feature_names, index=df.index)
                all_transformed.append(ohe_df)
                self.transformers_info.append(('one_hot_encoding', self._low_card_cats))
            else:
                self._ohe = None

            self._high_card_cats = self._get_high_cardinality_cats(df, self._cat_cols)
            if self._high_card_cats and y is not None:
                self._target_encoder = TargetEncoder(smoothing=self.target_encoder_smoothing)
                target_encoded = self._target_encoder.fit_transform(imputed_cat[self._high_card_cats], y)
                all_transformed.append(target_encoded)
                self.transformers_info.append(('target_encoding', self._high_card_cats))
            else:
                self._target_encoder = None

            self._freq_encoder = FrequencyEncoder()
            freq_encoded = self._freq_encoder.fit_transform(imputed_cat)
            all_transformed.append(freq_encoded)
            self.transformers_info.append(('frequency_encoding', self._cat_cols))

        if self._date_cols:
            date_df = df[self._date_cols].copy()
            self._date_extractor = DateFeatureExtractor()
            date_features = self._date_extractor.fit_transform(date_df)
            all_transformed.append(date_features)
            self.transformers_info.append(('date_extraction', self._date_cols))
        else:
            self._date_extractor = None

        if self._text_cols and self.text_strategy == 'tfidf':
            text_df = df[self._text_cols].copy()
            self._tfidf = TfidfTextTransformer(max_features=self.max_tfidf_features)
            tfidf_features = self._tfidf.fit_transform(text_df)
            all_transformed.append(tfidf_features)
            self.transformers_info.append(('tfidf', self._text_cols))
        else:
            self._tfidf = None

        if all_transformed:
            combined = pd.concat(all_transformed, axis=1)
        else:
            combined = pd.DataFrame(index=df.index)

        self._var_filter = VarianceFilter()
        combined = self._var_filter.fit_transform(combined)
        self.transformers_info.append(('variance_filter', f'removed {len(combined.columns) - combined.shape[1]} features'))

        if combined.shape[1] > 1:
            self._corr_filter = HighCorrelationFilter(threshold=self.corr_threshold)
            combined = self._corr_filter.fit_transform(combined)
            self.transformers_info.append(('high_corr_filter', f'threshold={self.corr_threshold}'))
        else:
            self._corr_filter = None

        self.feature_names_ = combined.columns.tolist()
        self._is_fitted = True
        return combined

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换新数据"""
        if not self._is_fitted:
            raise ValueError("请先调用fit_transform方法拟合数据")

        all_transformed = []

        if self._numeric_cols:
            numeric_df = df[self._numeric_cols].copy()

            imputed_numeric = pd.DataFrame(
                self._numeric_imputer.transform(numeric_df),
                columns=self._numeric_cols,
                index=df.index
            )

            scaled = pd.DataFrame(
                self._scaler.transform(imputed_numeric),
                columns=[f"{c}_scaled" for c in self._numeric_cols],
                index=df.index
            )
            all_transformed.append(scaled)

            binned = self._binner.transform(imputed_numeric)
            all_transformed.append(binned)

            if self._log_transformer is not None and self._log_cols:
                log_df = self._log_transformer.transform(imputed_numeric[self._log_cols])
                all_transformed.append(log_df)

            if self._missing_indicator is not None and self._missing_cols:
                missing_df = self._missing_indicator.transform(numeric_df[self._missing_cols])
                all_transformed.append(missing_df)

            if self._poly_cross is not None:
                poly_df = self._poly_cross.transform(imputed_numeric)
                all_transformed.append(poly_df)

        if self._cat_cols:
            cat_df = df[self._cat_cols].copy()
            imputed_cat = pd.DataFrame(
                self._cat_imputer.transform(cat_df),
                columns=self._cat_cols,
                index=df.index
            )

            if self._ohe is not None and self._low_card_cats:
                ohe_result = self._ohe.transform(imputed_cat[self._low_card_cats])
                ohe_df = pd.DataFrame(ohe_result, columns=self._ohe_feature_names, index=df.index)
                all_transformed.append(ohe_df)

            if self._target_encoder is not None and self._high_card_cats:
                target_encoded = self._target_encoder.transform(imputed_cat[self._high_card_cats])
                all_transformed.append(target_encoded)

            freq_encoded = self._freq_encoder.transform(imputed_cat)
            all_transformed.append(freq_encoded)

        if self._date_extractor is not None and self._date_cols:
            date_df = df[self._date_cols].copy()
            date_features = self._date_extractor.transform(date_df)
            all_transformed.append(date_features)

        if self._tfidf is not None and self._text_cols:
            text_df = df[self._text_cols].copy()
            tfidf_features = self._tfidf.transform(text_df)
            all_transformed.append(tfidf_features)

        if all_transformed:
            combined = pd.concat(all_transformed, axis=1)
        else:
            combined = pd.DataFrame(index=df.index)

        if self._var_filter is not None:
            combined = self._var_filter.transform(combined)

        if self._corr_filter is not None:
            combined = self._corr_filter.transform(combined)

        return combined

    def get_transform_info(self) -> List[Tuple[str, Any]]:
        """获取变换步骤信息"""
        return self.transformers_info

    def get_feature_count(self) -> int:
        """获取最终特征数量"""
        return len(self.feature_names_)

    def get_feature_names(self) -> List[str]:
        """获取最终特征名称"""
        return self.feature_names_.copy()
