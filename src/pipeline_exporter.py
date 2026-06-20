"""
Pipeline导出与复用模块
- pickle格式导出
- ONNX格式导出（如果支持）
- prediction_api.py模板生成
- model_card.md模型卡生成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pickle
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class PipelineExporter:
    """Pipeline导出器"""

    def __init__(
        self,
        preprocessor=None,
        feature_selector=None,
        model=None,
        column_types: Dict[str, str] = None,
        task_type: str = 'binary',
        feature_names: List[str] = None,
        target_column: str = '',
    ):
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.model = model
        self.column_types = column_types or {}
        self.task_type = task_type
        self.feature_names = feature_names or []
        self.target_column = target_column

        self.model_info = {}
        self.dataset_info = {}
        self.performance_metrics = {}
        self.feature_stats = {}

    def set_model_info(
        self,
        model_name: str,
        best_params: Dict,
        cv_mean: float,
        cv_std: float,
        train_time: float,
    ):
        """设置模型信息"""
        self.model_info = {
            'model_name': model_name,
            'best_params': best_params,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_time': train_time,
        }

    def set_dataset_info(
        self,
        n_rows: int,
        n_cols: int,
        n_features: int,
        feature_list: List[str],
    ):
        """设置数据集信息"""
        self.dataset_info = {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'n_features': n_features,
            'feature_list': feature_list,
        }

    def set_performance_metrics(self, metrics: Dict):
        """设置性能指标"""
        self.performance_metrics = metrics

    def export_pickle(self, output_path: str) -> str:
        """导出为pickle格式"""
        pipeline_data = {
            'version': '1.0.0',
            'task_type': self.task_type,
            'column_types': self.column_types,
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'selected_features': getattr(self.feature_selector, 'selected_features_', []),
            'model': self.model,
            'model_info': self.model_info,
            'dataset_info': self.dataset_info,
            'preprocessor': self.preprocessor,
            'feature_selector': self.feature_selector,
            'export_time': datetime.now().isoformat(),
        }

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(pipeline_data, f)

        return output_path

    def export_joblib(self, output_path: str) -> str:
        """导出为joblib格式（更高效）"""
        pipeline_data = {
            'version': '1.0.0',
            'task_type': self.task_type,
            'column_types': self.column_types,
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'selected_features': getattr(self.feature_selector, 'selected_features_', []),
            'model': self.model,
            'model_info': self.model_info,
            'dataset_info': self.dataset_info,
            'preprocessor': self.preprocessor,
            'feature_selector': self.feature_selector,
            'export_time': datetime.now().isoformat(),
        }

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        joblib.dump(pipeline_data, output_path)
        return output_path

    def export_onnx(self, output_path: str, X_sample: pd.DataFrame) -> Optional[str]:
        """导出为ONNX格式（如果支持）"""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            n_features = X_sample.shape[1]
            initial_type = [('float_input', FloatTensorType([None, n_features]))]

            onnx_model = convert_sklearn(self.model, initial_types=initial_type)

            dir_path = os.path.dirname(output_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

            return output_path
        except ImportError:
            return None
        except Exception as e:
            return None

    def generate_prediction_api(self, output_path: str) -> str:
        """生成prediction_api.py脚本模板"""
        template = '''"""
自动生成的预测API脚本
加载训练好的Pipeline，对新数据进行预测
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import warnings

warnings.filterwarnings("ignore")


class PredictionPipeline:
    """预测Pipeline类"""

    def __init__(self, model_path: str, use_joblib: bool = True):
        """
        加载训练好的Pipeline

        Args:
            model_path: 模型文件路径
            use_joblib: 是否使用joblib加载
        """
        if use_joblib:
            self.pipeline_data = joblib.load(model_path)
        else:
            with open(model_path, "rb") as f:
                self.pipeline_data = pickle.load(f)

        self.model = self.pipeline_data["model"]
        self.task_type = self.pipeline_data["task_type"]
        self.column_types = self.pipeline_data["column_types"]
        self.feature_names = self.pipeline_data["feature_names"]
        self.selected_features = self.pipeline_data.get("selected_features", [])
        self.target_column = self.pipeline_data.get("target_column", "")
        self.model_info = self.pipeline_data.get("model_info", {})

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        注意：此处为简化版本，实际使用时建议使用训练时完整的preprocessor对象
        """
        processed = df.copy()

        numeric_cols = [c for c, t in self.column_types.items() if t == "numeric"]
        cat_cols = [c for c, t in self.column_types.items() if t == "categorical"]

        for col in numeric_cols:
            if col in processed.columns:
                processed[col] = pd.to_numeric(processed[col], errors="coerce")
                processed[col] = processed[col].fillna(processed[col].median())

        for col in cat_cols:
            if col in processed.columns:
                mode_val = processed[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                processed[col] = processed[col].fillna(fill_val)

        return processed

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取特征数据"""
        feature_cols = self.selected_features if self.selected_features else self.feature_names
        available = [f for f in feature_cols if f in df.columns]

        if not available:
            available = [f for f in self.feature_names if f in df.columns]

        if not available:
            raise ValueError("没有可用的特征列，请检查输入数据")

        return df[available].copy()

    def predict(self, data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """
        对新数据进行预测

        Args:
            data: 输入数据，可以是DataFrame、字典或列表

        Returns:
            预测结果数组
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        processed = self._preprocess(df)
        X = self._get_features(processed)

        return self.model.predict(X)

    def predict_proba(self, data: Union[pd.DataFrame, Dict, List]) -> Optional[np.ndarray]:
        """
        预测概率（仅分类任务）

        Args:
            data: 输入数据

        Returns:
            概率矩阵或None（如果模型不支持）
        """
        if self.task_type not in ["binary", "multiclass"]:
            return None

        if not hasattr(self.model, "predict_proba"):
            return None

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        processed = self._preprocess(df)
        X = self._get_features(processed)

        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return self.model_info

    def get_task_type(self) -> str:
        """获取任务类型"""
        return self.task_type


def load_pipeline(model_path: str, use_joblib: bool = True) -> PredictionPipeline:
    """
    便捷函数：加载预测Pipeline

    Args:
        model_path: 模型文件路径
        use_joblib: 是否使用joblib加载

    Returns:
        PredictionPipeline实例
    """
    return PredictionPipeline(model_path, use_joblib=use_joblib)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python prediction_api.py <model_path> <input_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_csv = sys.argv[2]

    pipeline = load_pipeline(model_path)
    df = pd.read_csv(input_csv)
    predictions = pipeline.predict(df)

    print(f"预测完成，共 {len(predictions)} 条预测结果")
    print("前5条预测结果:")
    print(predictions[:5])
'''

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

        return output_path

    def generate_model_card(self, output_path: str) -> str:
        """生成model_card.md模型卡"""
        model_info = self.model_info
        dataset_info = self.dataset_info
        performance = self.performance_metrics

        task_type_cn = {
            'binary': '二分类',
            'multiclass': '多分类',
            'regression': '回归',
        }.get(self.task_type, self.task_type)

        lines = []
        lines.append('# 模型卡 (Model Card)')
        lines.append('')
        lines.append('## 基本信息')
        lines.append('')
        lines.append('| 项目 | 内容 |')
        lines.append('|------|------|')
        lines.append(f'| 模型名称 | {model_info.get("model_name", "未知")} |')
        lines.append(f'| 任务类型 | {task_type_cn} |')
        lines.append(f'| 目标列 | {self.target_column} |')
        lines.append(f'| 导出时间 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |')
        lines.append(f'| Pipeline版本 | 1.0.0 |')
        lines.append('')
        lines.append('## 数据集描述')
        lines.append('')
        lines.append(f'- 训练样本数: {dataset_info.get("n_rows", "未知")}')
        lines.append(f'- 原始特征数: {dataset_info.get("n_cols", "未知")}')
        lines.append(f'- 最终特征数: {len(self.feature_names)}')
        selected_count = len(getattr(self.feature_selector, 'selected_features_', []))
        lines.append(f'- 选中特征数: {selected_count}')
        lines.append('')
        lines.append('## 特征列表')
        lines.append('')

        if self.feature_names:
            lines.append('### 最终特征列表')
            lines.append('')
            for i, feat in enumerate(self.feature_names[:20], 1):
                lines.append(f'{i}. {feat}')
            if len(self.feature_names) > 20:
                lines.append('')
                lines.append(f'... 共 {len(self.feature_names)} 个特征')
        else:
            lines.append('（特征列表未提供）')

        lines.append('')
        lines.append('## 性能指标')
        lines.append('')

        if performance:
            for key, value in performance.items():
                if isinstance(value, (int, float)):
                    lines.append(f'- **{key}**: {round(value, 4)}')
                elif isinstance(value, str):
                    lines.append(f'- **{key}**: {value}')
        else:
            lines.append('（性能指标未提供）')

        lines.append('')
        lines.append('## 交叉验证结果')
        lines.append('')

        cv_mean = model_info.get('cv_mean', '未知')
        cv_std = model_info.get('cv_std', '未知')
        train_time = model_info.get('train_time', '未知')

        if isinstance(cv_mean, (int, float)):
            lines.append(f'- CV均值: {round(cv_mean, 4)}')
        else:
            lines.append(f'- CV均值: {cv_mean}')

        if isinstance(cv_std, (int, float)):
            lines.append(f'- CV标准差: {round(cv_std, 4)}')
        else:
            lines.append(f'- CV标准差: {cv_std}')

        if isinstance(train_time, (int, float)):
            lines.append(f'- 训练耗时: {round(train_time, 2)} 秒')
        else:
            lines.append(f'- 训练耗时: {train_time} 秒')

        lines.append('')
        lines.append('## 最佳超参数')
        lines.append('')

        best_params = model_info.get('best_params', {})
        if best_params:
            for key, value in best_params.items():
                lines.append(f'- **{key}**: {value}')
        else:
            lines.append('（超参数未提供）')

        lines.append('')
        lines.append('## 使用限制')
        lines.append('')
        lines.append('1. 本模型仅用于预测用途，不保证在所有场景下均适用')
        lines.append('2. 模型性能基于训练数据分布，数据分布变化时可能需要重新训练')
        lines.append('3. 请在使用前验证模型在实际数据上的表现')
        lines.append('4. 建议定期重新训练模型以保持性能')
        lines.append('5. 对于高风险决策场景请谨慎使用模型预测结果')
        lines.append('')
        lines.append('## 数据列类型说明')
        lines.append('')

        type_cn = {
            'numeric': '数值型',
            'categorical': '分类型',
            'date': '日期型',
            'text': '文本型',
            'id': 'ID型',
        }

        if self.column_types:
            for col, col_type in self.column_types.items():
                lines.append(f'- **{col}**: {type_cn.get(col_type, col_type)}')
        else:
            lines.append('（列类型信息未提供）')

        lines.append('')
        lines.append('---')
        lines.append('')
        lines.append('*此模型卡由 AutoML Pipeline 自动生成*')
        lines.append('')

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path

    def compute_feature_stats(self, X: pd.DataFrame) -> Dict:
        """计算训练数据的特征统计信息，用于漂移检测

        Args:
            X: 训练特征数据

        Returns:
            特征统计信息字典
        """
        stats = {}

        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [
            col for col in X.columns
            if col not in numeric_cols
        ]

        stats['numeric'] = {}
        for col in numeric_cols:
            series = X[col].dropna()
            if len(series) > 0:
                col_stats = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                    'n_samples': int(len(series)),
                    'missing_rate': float(X[col].isna().sum() / len(X)),
                }
                try:
                    from scipy.stats import skew, kurtosis
                    col_stats['skewness'] = float(skew(series))
                    col_stats['kurtosis'] = float(kurtosis(series))
                except Exception:
                    col_stats['skewness'] = 0.0
                    col_stats['kurtosis'] = 0.0
                stats['numeric'][col] = col_stats

        stats['categorical'] = {}
        for col in categorical_cols:
            series = X[col].dropna().astype(str)
            if len(series) > 0:
                value_counts = series.value_counts(normalize=True)
                stats['categorical'][col] = {
                    'n_unique': int(series.nunique()),
                    'top_value': str(value_counts.index[0]),
                    'top_frequency': float(value_counts.iloc[0]),
                    'n_samples': int(len(series)),
                    'missing_rate': float(X[col].isna().sum() / len(X)),
                    'distribution': {
                        str(k): float(v)
                        for k, v in value_counts.head(20).items()
                    }
                }

        self.feature_stats = stats
        return stats

    def generate_drift_detector(self, output_path: str) -> str:
        """生成数据漂移检测脚本 drift_detector.py

        Args:
            output_path: 输出文件路径

        Returns:
            文件路径
        """
        template = '''"""
自动生成的数据漂移检测脚本
加载训练时的特征统计信息，对新来的数据逐列做KS检验，
检测哪些特征发生了显著漂移（p值小于0.05），方便上线后监控。
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import json
import os

warnings.filterwarnings("ignore")

try:
    from scipy import stats
except ImportError:
    stats = None


class FeatureDriftDetector:
    """特征漂移检测器"""

    def __init__(self, stats_path: str, p_threshold: float = 0.05):
        """
        初始化漂移检测器

        Args:
            stats_path: 特征统计文件路径（.json 或 .pkl / .joblib）
            p_threshold: 显著性水平阈值，默认 0.05
        """
        self.p_threshold = p_threshold
        self.reference_stats = self._load_stats(stats_path)
        self.drift_results = {}

    def _load_stats(self, stats_path: str) -> Dict:
        """加载参考统计数据"""
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"统计文件不存在: {stats_path}")

        ext = os.path.splitext(stats_path)[1].lower()

        if ext == '.json':
            with open(stats_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.pkl':
            with open(stats_path, 'rb') as f:
                data = pickle.load(f)
                return data.get('feature_stats', data)
        elif ext == '.joblib':
            data = joblib.load(stats_path)
            return data.get('feature_stats', data)
        else:
            try:
                with open(stats_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

    def detect_numeric_drift(self, new_data: pd.DataFrame) -> List[Dict]:
        """
        检测数值型特征的漂移（KS检验）

        Args:
            new_data: 新数据 DataFrame

        Returns:
            漂移检测结果列表
        """
        if stats is None:
            print("警告: scipy 未安装，无法执行 KS 检验")
            return []

        results = []
        numeric_stats = self.reference_stats.get('numeric', {})

        for col, ref_stat in numeric_stats.items():
            if col not in new_data.columns:
                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'drift_detected': True,
                    'drift_type': 'missing_column',
                    'message': f'特征 {col} 在新数据中不存在',
                    'p_value': None,
                    'ks_statistic': None,
                })
                continue

            new_series = new_data[col].dropna()
            if len(new_series) < 2:
                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'drift_detected': False,
                    'drift_type': 'insufficient_data',
                    'message': f'特征 {col} 有效样本数不足',
                    'p_value': None,
                    'ks_statistic': None,
                })
                continue

            try:
                ref_mean = ref_stat.get('mean', 0)
                ref_std = ref_stat.get('std', 1) or 1

                ref_samples = np.random.normal(
                    ref_mean, ref_std,
                    size=max(len(new_series), 1000)
                )

                ks_stat, p_value = stats.ks_2samp(
                    ref_samples,
                    new_series.values
                )

                drift_detected = p_value < self.p_threshold

                new_mean = float(new_series.mean())
                new_std = float(new_series.std())

                mean_shift_pct = ((new_mean - ref_mean) / abs(ref_mean) * 100) if ref_mean != 0 else 0

                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'drift_detected': drift_detected,
                    'drift_type': 'distribution_shift' if drift_detected else 'no_drift',
                    'p_value': round(float(p_value), 6),
                    'ks_statistic': round(float(ks_stat), 6),
                    'ref_mean': round(ref_mean, 4),
                    'new_mean': round(new_mean, 4),
                    'mean_shift_pct': round(mean_shift_pct, 2),
                    'ref_std': round(ref_std, 4),
                    'new_std': round(new_std, 4),
                    'message': f'{"检测到漂移" if drift_detected else "无显著漂移"} (p={p_value:.4f})',
                })
            except Exception as e:
                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'drift_detected': False,
                    'drift_type': 'error',
                    'message': f'检测失败: {str(e)}',
                    'p_value': None,
                    'ks_statistic': None,
                })

        return results

    def detect_categorical_drift(self, new_data: pd.DataFrame) -> List[Dict]:
        """
        检测分类型特征的漂移（卡方检验 / PSI）

        Args:
            new_data: 新数据 DataFrame

        Returns:
            漂移检测结果列表
        """
        results = []
        cat_stats = self.reference_stats.get('categorical', {})

        for col, ref_stat in cat_stats.items():
            if col not in new_data.columns:
                results.append({
                    'feature': col,
                    'type': 'categorical',
                    'drift_detected': True,
                    'drift_type': 'missing_column',
                    'message': f'特征 {col} 在新数据中不存在',
                })
                continue

            new_series = new_data[col].dropna().astype(str)
            if len(new_series) == 0:
                results.append({
                    'feature': col,
                    'type': 'categorical',
                    'drift_detected': True,
                    'drift_type': 'all_missing',
                    'message': f'特征 {col} 全部为缺失值',
                })
                continue

            ref_dist = ref_stat.get('distribution', {})
            new_dist = new_series.value_counts(normalize=True).to_dict()

            psi = self._calculate_psi(ref_dist, new_dist)

            drift_detected = psi > 0.25

            results.append({
                'feature': col,
                'type': 'categorical',
                'drift_detected': drift_detected,
                'drift_type': 'distribution_shift' if drift_detected else 'no_drift',
                'psi': round(psi, 6),
                'ref_top_value': ref_stat.get('top_value', ''),
                'new_top_value': max(new_dist, key=new_dist.get) if new_dist else '',
                'message': self._psi_interpretation(psi),
            })

        return results

    def _calculate_psi(self, expected: Dict, actual: Dict) -> float:
        """计算 PSI (Population Stability Index)"""
        all_categories = set(expected.keys()) | set(actual.keys())
        if not all_categories:
            return 0.0

        epsilon = 1e-6
        psi = 0.0

        for cat in all_categories:
            e = expected.get(cat, epsilon / 2)
            a = actual.get(cat, epsilon / 2)

            if e == 0:
                e = epsilon
            if a == 0:
                a = epsilon

            psi += (e - a) * np.log(e / a)

        return float(psi)

    def _psi_interpretation(self, psi: float) -> str:
        """PSI 结果解释"""
        if psi < 0.1:
            return f'无显著漂移 (PSI={psi:.4f})'
        elif psi < 0.25:
            return f'轻度漂移 (PSI={psi:.4f})'
        else:
            return f'显著漂移 (PSI={psi:.4f})'

    def detect_all_drift(self, new_data: pd.DataFrame) -> Dict:
        """
        检测所有特征的漂移

        Args:
            new_data: 新数据 DataFrame

        Returns:
            完整的漂移检测结果
        """
        numeric_results = self.detect_numeric_drift(new_data)
        categorical_results = self.detect_categorical_drift(new_data)

        all_results = numeric_results + categorical_results

        drifted_features = [
            r for r in all_results
            if r.get('drift_detected', False)
        ]

        summary = {
            'total_features': len(all_results),
            'drifted_count': len(drifted_features),
            'drifted_ratio': round(len(drifted_features) / max(len(all_results), 1), 4),
            'numeric_drifted': sum(1 for r in numeric_results if r.get('drift_detected')),
            'categorical_drifted': sum(1 for r in categorical_results if r.get('drift_detected')),
            'drifted_features': [r['feature'] for r in drifted_features],
        }

        self.drift_results = {
            'summary': summary,
            'numeric': numeric_results,
            'categorical': categorical_results,
            'all': all_results,
        }

        return self.drift_results

    def print_report(self):
        """打印漂移检测报告"""
        if not self.drift_results:
            print("请先调用 detect_all_drift() 方法")
            return

        summary = self.drift_results['summary']

        print('=' * 60)
        print('数据漂移检测报告')
        print('=' * 60)
        print(f'总特征数: {summary["total_features"]}')
        print(f'漂移特征数: {summary["drifted_count"]}')
        print(f'漂移比例: {summary["drifted_ratio"] * 100:.2f}%')
        print(f'数值型漂移: {summary["numeric_drifted"]}')
        print(f'分类型漂移: {summary["categorical_drifted"]}')
        print()

        if summary['drifted_features']:
            print('⚠️  发生漂移的特征:')
            for feat in summary['drifted_features']:
                print(f'   - {feat}')
            print()
        else:
            print('✅ 未检测到显著漂移')
            print()

        print('-' * 60)
        print('详细结果:')
        print('-' * 60)

        for result in self.drift_results['all']:
            status = '❌' if result.get('drift_detected') else '✅'
            print(f'{status} {result["feature"]} ({result["type"]}): {result.get("message", "")}')

        print('=' * 60)

    def to_dataframe(self) -> pd.DataFrame:
        """将检测结果转换为 DataFrame"""
        if not self.drift_results:
            return pd.DataFrame()

        return pd.DataFrame(self.drift_results['all'])


def main():
    """命令行入口"""
    import sys

    if len(sys.argv) < 3:
        print("用法: python drift_detector.py <stats_file> <new_data_csv>")
        print()
        print("示例:")
        print("  python drift_detector.py feature_stats.json new_data.csv")
        sys.exit(1)

    stats_path = sys.argv[1]
    data_path = sys.argv[2]

    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)

    try:
        new_data = pd.read_csv(data_path)
    except Exception as e:
        print(f"读取数据失败: {e}")
        sys.exit(1)

    detector = FeatureDriftDetector(stats_path)

    print(f"加载新数据: {len(new_data)} 行, {len(new_data.columns)} 列")
    print()

    result = detector.detect_all_drift(new_data)
    detector.print_report()

    return result


if __name__ == "__main__":
    main()
'''

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

        return output_path

    def export_feature_stats(self, output_path: str) -> str:
        """导出特征统计信息为 JSON 格式

        Args:
            output_path: 输出文件路径

        Returns:
            文件路径
        """
        if not self.feature_stats:
            return ''

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_stats, f, indent=2, ensure_ascii=False)

        return output_path

    def export_all(
        self,
        output_dir: str,
        X_sample: Optional[pd.DataFrame] = None,
        include_onnx: bool = True,
    ) -> Dict[str, str]:
        """导出所有文件"""
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        pickle_path = os.path.join(output_dir, 'pipeline.pkl')
        self.export_pickle(pickle_path)
        results['pickle'] = pickle_path

        joblib_path = os.path.join(output_dir, 'pipeline.joblib')
        self.export_joblib(joblib_path)
        results['joblib'] = joblib_path

        api_path = os.path.join(output_dir, 'prediction_api.py')
        self.generate_prediction_api(api_path)
        results['prediction_api'] = api_path

        card_path = os.path.join(output_dir, 'model_card.md')
        self.generate_model_card(card_path)
        results['model_card'] = card_path

        if X_sample is not None and hasattr(X_sample, 'columns') and len(X_sample.columns) > 0:
            self.compute_feature_stats(X_sample)
            stats_path = os.path.join(output_dir, 'feature_stats.json')
            self.export_feature_stats(stats_path)
            results['feature_stats'] = stats_path

            drift_path = os.path.join(output_dir, 'drift_detector.py')
            self.generate_drift_detector(drift_path)
            results['drift_detector'] = drift_path

        if include_onnx and X_sample is not None:
            onnx_path = os.path.join(output_dir, 'model.onnx')
            onnx_result = self.export_onnx(onnx_path, X_sample)
            if onnx_result:
                results['onnx'] = onnx_result

        return results
