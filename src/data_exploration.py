"""
数据接入与探索模块
- CSV文件加载
- 数据类型推断
- 数据概况报告生成
- 数据集采样
- 目标列校验
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from scipy import stats
from sklearn.model_selection import train_test_split


class DataTypeInference:
    """数据类型推断器"""

    NUMERIC_TYPES = ['int64', 'float64', 'int32', 'float32']
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{4}/\d{2}/\d{2}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{4}年\d{1,2}月\d{1,2}日',
    ]

    @classmethod
    def infer_type(cls, series: pd.Series) -> str:
        """推断单列数据类型: 数值型/分类型/日期型/文本型/ID型"""
        name = series.name
        n_unique = series.nunique(dropna=True)
        n_total = len(series)
        dtype = str(series.dtype)

        if dtype in cls.NUMERIC_TYPES:
            if n_unique <= 2 and n_total > 10:
                return 'categorical'
            if n_unique == n_total and n_total > 20:
                if 'id' in str(name).lower() or '编号' in str(name) or 'ID' in str(name):
                    return 'id'
            return 'numeric'

        if dtype == 'datetime64[ns]' or dtype == 'datetime64[ns, UTC]':
            return 'date'

        if dtype == 'object' or dtype == 'string':
            non_null = series.dropna().astype(str)
            if len(non_null) == 0:
                return 'categorical'

            date_match_count = 0
            for val in non_null.head(20):
                for pattern in cls.DATE_PATTERNS:
                    if re.match(pattern, val):
                        date_match_count += 1
                        break

            if date_match_count >= len(non_null.head(20)) * 0.8 and len(non_null.head(20)) > 0:
                return 'date'

            if n_unique <= 10 and n_unique / max(n_total, 1) < 0.2:
                return 'categorical'

            if n_unique == n_total and n_total > 20:
                if 'id' in str(name).lower() or '编号' in str(name) or 'ID' in str(name):
                    return 'id'

            avg_len = non_null.str.len().mean()
            if avg_len > 30 or n_unique > 50:
                return 'text'

            return 'categorical'

        if dtype == 'bool':
            return 'categorical'

        return 'categorical'

    @classmethod
    def infer_all(cls, df: pd.DataFrame) -> Dict[str, str]:
        """推断所有列的数据类型"""
        types = {}
        for col in df.columns:
            types[col] = cls.infer_type(df[col])
        return types


class DataExplorer:
    """数据探索器"""

    def __init__(self, df: pd.DataFrame, column_types: Dict[str, str]):
        self.df = df
        self.column_types = column_types
        self.n_rows, self.n_cols = df.shape

    def get_overview(self) -> Dict:
        """获取数据概况"""
        return {
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'memory_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'n_numeric': sum(1 for t in self.column_types.values() if t == 'numeric'),
            'n_categorical': sum(1 for t in self.column_types.values() if t == 'categorical'),
            'n_date': sum(1 for t in self.column_types.values() if t == 'date'),
            'n_text': sum(1 for t in self.column_types.values() if t == 'text'),
            'n_id': sum(1 for t in self.column_types.values() if t == 'id'),
        }

    def get_column_stats(self, col: str) -> Dict:
        """获取单列统计信息"""
        series = self.df[col]
        col_type = self.column_types[col]
        n_missing = series.isna().sum()
        missing_rate = round(n_missing / len(series) * 100, 2)
        n_unique = series.nunique(dropna=True)

        stats_dict = {
            'column': col,
            'type': col_type,
            'missing_count': n_missing,
            'missing_rate': missing_rate,
            'unique_count': n_unique,
            'unique_rate': round(n_unique / len(series) * 100, 2),
        }

        if col_type == 'numeric':
            numeric_stats = self._get_numeric_stats(series)
            stats_dict.update(numeric_stats)
        elif col_type == 'categorical':
            cat_stats = self._get_categorical_stats(series)
            stats_dict.update(cat_stats)
        elif col_type == 'date':
            date_stats = self._get_date_stats(series)
            stats_dict.update(date_stats)
        elif col_type == 'text':
            text_stats = self._get_text_stats(series)
            stats_dict.update(text_stats)

        return stats_dict

    def _get_numeric_stats(self, series: pd.Series) -> Dict:
        """数值型列统计"""
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        return {
            'mean': round(clean.mean(), 4),
            'median': round(clean.median(), 4),
            'std': round(clean.std(), 4),
            'min': round(clean.min(), 4),
            'max': round(clean.max(), 4),
            'q25': round(clean.quantile(0.25), 4),
            'q75': round(clean.quantile(0.75), 4),
            'skewness': round(stats.skew(clean), 4),
            'kurtosis': round(stats.kurtosis(clean), 4),
            'all_positive': (clean > 0).all(),
        }

    def _get_categorical_stats(self, series: pd.Series) -> Dict:
        """分类型列统计"""
        value_counts = series.dropna().value_counts().head(10)
        return {
            'top10_values': [
                {'value': str(v), 'count': int(c), 'percentage': round(c / len(series.dropna()) * 100, 2)}
                for v, c in value_counts.items()
            ]
        }

    def _get_date_stats(self, series: pd.Series) -> Dict:
        """日期型列统计"""
        try:
            dt_series = pd.to_datetime(series.dropna(), errors='coerce').dropna()
            if len(dt_series) == 0:
                return {}
            return {
                'min_date': str(dt_series.min().date()),
                'max_date': str(dt_series.max().date()),
                'range_days': (dt_series.max() - dt_series.min()).days,
            }
        except:
            return {}

    def _get_text_stats(self, series: pd.Series) -> Dict:
        """文本型列统计"""
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}
        lengths = clean.str.len()
        return {
            'avg_length': round(lengths.mean(), 2),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
        }

    def get_all_column_stats(self) -> List[Dict]:
        """获取所有列的统计信息"""
        return [self.get_column_stats(col) for col in self.df.columns]

    def get_correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """计算Pearson相关矩阵"""
        if columns is None:
            numeric_cols = [c for c, t in self.column_types.items() if t == 'numeric']
        else:
            numeric_cols = columns

        if len(numeric_cols) < 2:
            return pd.DataFrame()

        numeric_df = self.df[numeric_cols].select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()

        return numeric_df.corr()


class DataSampler:
    """数据采样器"""

    @staticmethod
    def stratified_sample(
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        sample_size: int = 10000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """分层采样，保持目标列分布一致"""
        if len(df) <= sample_size:
            return df.copy()

        if task_type in ['binary', 'multiclass']:
            try:
                _, sample_df = train_test_split(
                    df,
                    test_size=min(sample_size / len(df), 0.99),
                    stratify=df[target_col],
                    random_state=random_state
                )
                return sample_df.reset_index(drop=True)
            except:
                return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        else:
            return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


class TargetValidator:
    """目标列校验器"""

    @staticmethod
    def validate(
        df: pd.DataFrame,
        target_col: str,
        task_type: str
    ) -> Tuple[bool, str]:
        """校验目标列合法性

        Returns:
            (是否合法, 错误信息)
        """
        if target_col not in df.columns:
            return False, f"目标列 '{target_col}' 不存在于数据集中"

        series = df[target_col]

        if series.isna().all():
            return False, "目标列全部为缺失值"

        if task_type in ['binary', 'multiclass']:
            n_classes = series.dropna().nunique()
            if n_classes < 2:
                return False, f"分类任务需要至少2个类别，当前目标列只有 {n_classes} 个类别"
            if n_classes > 50:
                return False, f"分类任务类别数不能超过50，当前目标列有 {n_classes} 个类别"
            if task_type == 'binary' and n_classes != 2:
                return False, f"二分类任务需要恰好2个类别，当前有 {n_classes} 个类别"
            return True, "目标列校验通过"

        elif task_type == 'regression':
            if str(series.dtype) not in ['int64', 'float64', 'int32', 'float32']:
                try:
                    pd.to_numeric(series, errors='raise')
                    return True, "目标列校验通过（可转换为数值型）"
                except:
                    return False, "回归任务目标列必须是数值型"
            return True, "目标列校验通过"

        return False, f"未知的任务类型: {task_type}"


def load_csv(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """加载CSV文件，自动尝试多种编码"""
    encodings = [encoding, 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    raise ValueError("无法读取CSV文件，请检查文件编码格式")


class DataQualityScorer:
    """数据质量评分器"""

    GRADE_EXCELLENT = '优秀'
    GRADE_GOOD = '良好'
    GRADE_FAIR = '一般'
    GRADE_POOR = '较差'

    def __init__(self, df: pd.DataFrame, column_types: Dict[str, str]):
        self.df = df
        self.column_types = column_types
        self.n_rows = len(df)
        self.n_cols = len(df.columns)

    def _score_missing(self) -> Dict:
        """缺失率评分"""
        total_cells = self.n_rows * self.n_cols
        missing_cells = self.df.isna().sum().sum()
        missing_rate = missing_cells / total_cells * 100 if total_cells > 0 else 0

        if missing_rate < 5:
            score = 100 - missing_rate * 2
        elif missing_rate < 20:
            score = 90 - (missing_rate - 5) * 2
        else:
            score = max(0, 60 - (missing_rate - 20) * 1.5)

        score = round(max(0, min(100, score)), 2)

        return {
            'score': score,
            'missing_rate': round(missing_rate, 2),
            'missing_cells': int(missing_cells),
            'total_cells': total_cells,
        }

    def _score_duplicates(self) -> Dict:
        """重复行比例评分"""
        n_duplicates = self.df.duplicated().sum()
        duplicate_rate = n_duplicates / self.n_rows * 100 if self.n_rows > 0 else 0

        if duplicate_rate < 5:
            score = 100 - duplicate_rate * 2
        elif duplicate_rate < 20:
            score = 90 - (duplicate_rate - 5) * 2
        else:
            score = max(0, 60 - (duplicate_rate - 20) * 1.5)

        score = round(max(0, min(100, score)), 2)

        return {
            'score': score,
            'duplicate_rate': round(duplicate_rate, 2),
            'n_duplicates': int(n_duplicates),
            'n_rows': self.n_rows,
        }

    def _score_outliers(self) -> Dict:
        """异常值比例评分（IQR方法）"""
        numeric_cols = [c for c, t in self.column_types.items() if t == 'numeric']
        n_numeric = len(numeric_cols)

        if n_numeric == 0:
            return {
                'score': 100,
                'outlier_cols_ratio': 0,
                'n_outlier_cols': 0,
                'n_numeric_cols': 0,
                'per_col_outliers': {},
            }

        outlier_cols = 0
        per_col_outliers = {}

        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                per_col_outliers[col] = 0
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                per_col_outliers[col] = 0
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            n_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_ratio = n_outliers / len(series) * 100
            per_col_outliers[col] = round(outlier_ratio, 2)

            if outlier_ratio > 10:
                outlier_cols += 1

        outlier_cols_ratio = outlier_cols / n_numeric * 100

        if outlier_cols_ratio < 10:
            score = 100 - outlier_cols_ratio * 1
        elif outlier_cols_ratio < 30:
            score = 90 - (outlier_cols_ratio - 10) * 1.5
        else:
            score = max(0, 60 - (outlier_cols_ratio - 30) * 1)

        score = round(max(0, min(100, score)), 2)

        return {
            'score': score,
            'outlier_cols_ratio': round(outlier_cols_ratio, 2),
            'n_outlier_cols': outlier_cols,
            'n_numeric_cols': n_numeric,
            'per_col_outliers': per_col_outliers,
        }

    def calculate_score(self) -> Dict:
        """计算综合数据质量评分"""
        missing_result = self._score_missing()
        duplicate_result = self._score_duplicates()
        outlier_result = self._score_outliers()

        overall_score = round(
            (missing_result['score'] * 0.4 +
             duplicate_result['score'] * 0.3 +
             outlier_result['score'] * 0.3),
            2
        )

        if overall_score >= 90:
            grade = self.GRADE_EXCELLENT
        elif overall_score >= 75:
            grade = self.GRADE_GOOD
        elif overall_score >= 60:
            grade = self.GRADE_FAIR
        else:
            grade = self.GRADE_POOR

        return {
            'overall_score': overall_score,
            'grade': grade,
            'missing': missing_result,
            'duplicates': duplicate_result,
            'outliers': outlier_result,
        }
