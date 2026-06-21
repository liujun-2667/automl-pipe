"""
AutoML Pipeline 主模块
整合数据探索、特征工程、特征选择、模型选择、诊断、导出等所有功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import warnings
import time

from src.data_exploration import (
    DataTypeInference, DataExplorer, DataSampler, TargetValidator, load_csv
)
from src.feature_engineering import AutoFeatureEngineer
from src.feature_selection import FeatureImportanceAnalyzer, IntersectionFeatureSelector
from src.model_selector import AutoModelSelector
from src.model_diagnosis import ModelDiagnostician
from src.pipeline_exporter import PipelineExporter
from src.interpretability import ModelInterpretabilityAnalyzer
from src.drift_detection import DriftDetector, AlertStorage

warnings.filterwarnings('ignore')


class AutoMLPipeline:
    """AutoML Pipeline 主类"""

    def __init__(
        self,
        task_type: str = 'binary',
        target_column: str = '',
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.target_column = target_column
        self.random_state = random_state

        self.df = None
        self.df_fast = None
        self.column_types = {}

        self.explorer = None
        self.feature_engineer = None
        self.feature_analyzer = None
        self.feature_selector = None
        self.model_selector = None
        self.diagnostician = None
        self.interpretability_analyzer = None
        self.drift_detector = None
        self.alert_storage = None
        self.exporter = None

        self.X_full = None
        self.y_full = None
        self.X_fast = None
        self.y_fast = None
        self.X_selected = None
        self.X_fast_selected = None

        self.is_data_loaded = False
        self.is_feature_engineered = False
        self.is_feature_selected = False
        self.is_model_trained = False
        self.is_drift_detected = False

    def load_data(self, df: pd.DataFrame) -> Dict:
        """加载数据并初始化"""
        self.df = df

        self.column_types = DataTypeInference.infer_all(df)

        self.explorer = DataExplorer(df, self.column_types)

        self.is_data_loaded = True

        return {
            'overview': self.explorer.get_overview(),
            'column_types': self.column_types,
        }

    def update_column_type(self, column: str, new_type: str):
        """手动更新列类型"""
        if column in self.column_types:
            self.column_types[column] = new_type
            self.explorer = DataExplorer(self.df, self.column_types)

    def validate_target(self, target_col: str, task_type: str) -> tuple[bool, str]:
        """验证目标列"""
        self.target_column = target_col
        self.task_type = task_type
        return TargetValidator.validate(self.df, target_col, task_type)

    def prepare_datasets(self, sample_size: int = 10000) -> Dict:
        """准备完整数据集和快速实验子集"""
        if len(self.df) > 100000:
            self.df_fast = DataSampler.stratified_sample(
                self.df, self.target_column, self.task_type,
                sample_size=sample_size,
                random_state=self.random_state
            )
            use_sampling = True
        else:
            self.df_fast = self.df.copy()
            use_sampling = False

        y_full = self.df[self.target_column].copy()
        X_full = self.df.drop(columns=[self.target_column])

        y_fast = self.df_fast[self.target_column].copy()
        X_fast = self.df_fast.drop(columns=[self.target_column])

        self.y_full = y_full
        self.X_full_input = X_full
        self.y_fast = y_fast
        self.X_fast_input = X_fast

        return {
            'full_rows': len(self.df),
            'fast_rows': len(self.df_fast),
            'use_sampling': use_sampling,
        }

    def run_feature_engineering(
        self,
        text_strategy: str = 'tfidf',
        enable_poly_cross: bool = True,
        corr_threshold: float = 0.95,
        max_tfidf_features: int = 100,
        n_bins: int = 5,
    ) -> Dict:
        """运行自动特征工程"""
        feature_column_types = {
            col: col_type
            for col, col_type in self.column_types.items()
            if col != self.target_column
        }

        numeric_count = sum(1 for t in feature_column_types.values() if t == 'numeric')

        self.feature_engineer = AutoFeatureEngineer(
            column_types=feature_column_types,
            task_type=self.task_type,
            text_strategy=text_strategy,
            enable_poly_cross=enable_poly_cross and numeric_count < 20,
            corr_threshold=corr_threshold,
            max_tfidf_features=max_tfidf_features,
            n_bins=n_bins,
        )

        X_full_transformed = self.feature_engineer.fit_transform(
            self.X_full_input, self.y_full
        )

        X_fast_transformed = self.feature_engineer.transform(self.X_fast_input)

        self.X_full = X_full_transformed
        self.X_fast = X_fast_transformed

        self.is_feature_engineered = True

        return {
            'n_features_full': X_full_transformed.shape[1],
            'n_features_fast': X_fast_transformed.shape[1],
            'transform_steps': self.feature_engineer.get_transform_info(),
            'feature_names': self.feature_engineer.get_feature_names(),
        }

    def run_feature_selection(
        self,
        n_methods_required: int = 2,
        n_estimators: int = 100,
        n_features: Optional[int] = None,
        auto: bool = True,
        auto_threshold: float = 0.8,
    ) -> Dict:
        """运行特征选择"""
        self.feature_analyzer = FeatureImportanceAnalyzer(
            task_type=self.task_type,
            n_estimators=n_estimators,
            random_state=self.random_state,
        )
        self.feature_analyzer.fit(self.X_fast, self.y_fast)

        self.feature_selector = IntersectionFeatureSelector(
            n_methods_required=n_methods_required,
            auto_threshold=auto_threshold,
        )
        self.feature_selector.fit(
            self.feature_analyzer,
            n_features=n_features,
            auto=auto,
        )

        self.X_selected = self.feature_selector.transform(self.X_full)
        self.X_fast_selected = self.feature_selector.transform(self.X_fast)

        self.is_feature_selected = True

        return {
            'n_selected': self.feature_selector.get_selected_count(),
            'selected_features': self.feature_selector.selected_features_,
            'importances': self.feature_analyzer.get_all_importances(),
        }

    def run_model_selection(
        self,
        n_trials: int = 30,
        cv: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """运行自动模型选择"""
        self.model_selector = AutoModelSelector(
            task_type=self.task_type,
            n_trials=n_trials,
            cv=cv,
            random_state=self.random_state,
        )

        results_df = self.model_selector.fit(
            self.X_fast_selected if self.is_feature_selected else self.X_fast,
            self.y_fast,
            self.X_selected if self.is_feature_selected else self.X_full,
            self.y_full,
            progress_callback=progress_callback,
        )

        self.is_model_trained = True

        return {
            'results_df': results_df,
            'best_model_name': self.model_selector.get_best_model_name(),
            'best_score': self.model_selector.get_best_score(),
            'best_model': self.model_selector.get_best_model(),
        }

    def stop_model_selection(self):
        """停止模型搜索"""
        if self.model_selector:
            self.model_selector.stop()

    def run_diagnosis(self) -> Dict:
        """运行模型诊断"""
        X = self.X_selected if self.is_feature_selected else self.X_full
        y = self.y_full
        best_model = self.model_selector.get_best_model()

        self.diagnostician = ModelDiagnostician(
            task_type=self.task_type,
            random_state=self.random_state,
        )
        self.diagnostician.split_data(X, y)
        metrics = self.diagnostician.evaluate_model(best_model)

        shap_data = self.diagnostician.get_shap_values(best_model, X)

        if self.task_type == 'binary':
            lc_scoring = 'roc_auc'
        elif self.task_type == 'multiclass':
            lc_scoring = 'f1_macro'
        else:
            lc_scoring = 'neg_root_mean_squared_error'

        learning_curve_data = self.diagnostician.learning_curve_analysis(
            best_model, X, y, cv=3, scoring=lc_scoring
        )

        return {
            'metrics': metrics,
            'shap_data': shap_data,
            'is_overfitting': metrics.get('is_overfitting', False),
            'overfitting_suggestions': self.diagnostician.get_overfitting_suggestions() if metrics.get('is_overfitting', False) else [],
            'learning_curve': learning_curve_data,
        }

    def run_interpretability(self, output_dir: str = './output') -> Dict:
        """运行模型可解释性分析"""
        X = self.X_selected if self.is_feature_selected else self.X_full
        y = self.y_full
        best_model = self.model_selector.get_best_model()
        best_model_name = self.model_selector.get_best_model_name() if self.model_selector else 'Unknown'
        feature_names = list(X.columns) if X is not None else []

        all_models = None
        if self.model_selector:
            top_models_raw = self.model_selector.get_top_models(n=3)
            all_models = [(name, model) for name, model, _ in top_models_raw]

        self.interpretability_analyzer = ModelInterpretabilityAnalyzer(
            model=best_model,
            X=X,
            y=y,
            task_type=self.task_type,
            model_name=best_model_name,
            feature_names=feature_names,
            column_types=self.column_types,
            random_state=self.random_state,
            all_models=all_models,
        )

        result = self.interpretability_analyzer.run_full_analysis(output_dir=output_dir)

        return result

    def run_drift_detection(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        new_data: Optional[pd.DataFrame] = None,
        storage_path: str = './drift_alerts.json',
        dataset_name: str = 'unknown',
        save_alert: bool = True,
    ) -> Dict:
        """运行数据漂移检测

        Args:
            reference_data: 参考数据集(通常是训练时的验证集)。如果为None则尝试使用diagnostician的X_test
            new_data: 待检测的新数据集。如果为None且reference_data已指定，则用validation集作为新数据做基线
            storage_path: 告警持久化存储的JSON文件路径
            dataset_name: 数据集名称(用于存储告警记录)
            save_alert: 是否将检测结果持久化存储

        Returns:
            完整的漂移检测结果字典
        """
        if reference_data is None:
            if self.diagnostician and hasattr(self.diagnostician, 'X_train') and self.diagnostician.X_train is not None:
                reference_data = self.diagnostician.X_train
            elif self.X_full is not None:
                reference_data = self.X_full

        if reference_data is None:
            return {'error': '未提供参考数据集，且无法从Pipeline内部获取'}

        feature_column_types = {
            col: col_type
            for col, col_type in self.column_types.items()
            if col != self.target_column
        }

        ref_feature_cols = [
            c for c in reference_data.columns
            if c in feature_column_types and feature_column_types[c] in ('numeric', 'categorical')
        ]
        reference_data_filtered = reference_data[ref_feature_cols].copy()

        self.drift_detector = DriftDetector(
            reference_data=reference_data_filtered,
            column_types=feature_column_types,
        )

        if new_data is None:
            if self.diagnostician and hasattr(self.diagnostician, 'X_test') and self.diagnostician.X_test is not None:
                new_data = self.diagnostician.X_test
            elif self.X_full is not None:
                new_data = self.X_full

        if new_data is None:
            return {'error': '未提供待检测数据集，且无法从Pipeline内部获取'}

        new_feature_cols = [c for c in ref_feature_cols if c in new_data.columns]
        new_data_filtered = new_data[new_feature_cols].copy()

        detection_result = self.drift_detector.detect(new_data_filtered)

        if save_alert:
            try:
                self.alert_storage = AlertStorage(storage_path=storage_path)
                self.alert_storage.save_alert(detection_result, dataset_name=dataset_name)
            except Exception:
                pass

        self.is_drift_detected = True
        self._last_drift_result = detection_result

        return detection_result

    def export_pipeline(
        self,
        output_dir: str,
        include_onnx: bool = True,
    ) -> Dict[str, str]:
        """导出完整Pipeline"""
        X_sample = self.X_selected if self.is_feature_selected else self.X_full
        if X_sample is not None and len(X_sample) > 0:
            X_sample = X_sample.head(100)

        best_model_name = self.model_selector.get_best_model_name() if self.model_selector else ''
        best_model = self.model_selector.get_best_model() if self.model_selector else None
        best_params = self.model_selector.best_params_ if self.model_selector else {}
        best_score = self.model_selector.best_score_ if self.model_selector else 0.0

        results_df = self.model_selector.get_results_df() if self.model_selector else pd.DataFrame()
        if not results_df.empty:
            best_row = results_df[results_df['model_name'] == best_model_name]
            if not best_row.empty:
                cv_std = best_row.iloc[0].get('cv_std', 0.0)
                train_time = best_row.iloc[0].get('train_time', 0.0)
            else:
                cv_std = 0.0
                train_time = 0.0
        else:
            cv_std = 0.0
            train_time = 0.0

        self.exporter = PipelineExporter(
            preprocessor=self.feature_engineer,
            feature_selector=self.feature_selector,
            model=best_model,
            column_types=self.column_types,
            task_type=self.task_type,
            feature_names=self.feature_engineer.get_feature_names() if self.feature_engineer else [],
            target_column=self.target_column,
        )

        self.exporter.set_model_info(
            model_name=best_model_name,
            best_params=best_params,
            cv_mean=best_score,
            cv_std=cv_std,
            train_time=train_time,
        )

        self.exporter.set_dataset_info(
            n_rows=len(self.df) if self.df is not None else 0,
            n_cols=len(self.column_types),
            n_features=X_sample.shape[1] if X_sample is not None else 0,
            feature_list=self.feature_engineer.get_feature_names() if self.feature_engineer else [],
        )

        if self.diagnostician and hasattr(self.diagnostician, 'y_test') and self.diagnostician.y_test is not None:
            metrics = self.diagnostician.evaluate_model(best_model)
            self.exporter.set_performance_metrics(metrics)

        results = self.exporter.export_all(
            output_dir=output_dir,
            X_sample=X_sample,
            include_onnx=include_onnx,
        )

        return results

    def get_progress(self) -> Dict:
        """获取模型搜索进度"""
        if self.model_selector:
            return self.model_selector.get_progress()
        return {}

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = 'binary',
        sample_size: int = 10000,
        text_strategy: str = 'tfidf',
        enable_poly_cross: bool = True,
        corr_threshold: float = 0.95,
        max_tfidf_features: int = 100,
        n_bins: int = 5,
        n_methods_required: int = 2,
        n_estimators: int = 100,
        n_features: Optional[int] = None,
        auto: bool = True,
        auto_threshold: float = 0.8,
        n_trials: int = 30,
        cv: int = 5,
        interpretability_output_dir: str = './output',
        enable_drift_detection: bool = True,
        drift_reference_data: Optional[pd.DataFrame] = None,
        drift_storage_path: str = './drift_alerts.json',
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """一站式运行完整Pipeline

        Args:
            df: 输入数据集
            target_column: 目标列名
            task_type: 任务类型 binary/multiclass/regression
            sample_size: 快速实验子集大小
            text_strategy: 文本处理策略
            enable_poly_cross: 是否启用多项式交叉
            corr_threshold: 高相关过滤阈值
            max_tfidf_features: TF-IDF最大特征数
            n_bins: 数值分箱档数
            n_methods_required: 特征选择最少方法数
            n_estimators: 随机森林树数量
            n_features: 手动指定保留特征数
            auto: 自动选择特征数
            auto_threshold: 累计重要性阈值
            n_trials: 每个模型试验次数
            cv: 交叉验证折数
            interpretability_output_dir: 可解释性输出目录
            enable_drift_detection: 是否启用漂移检测
            drift_reference_data: 漂移检测参考数据集(通常为训练集)。
                如果传入了该参数，训练结束后会自动运行一次漂移检测(用验证集作为新数据)。
            drift_storage_path: 告警持久化存储路径
            progress_callback: 进度回调函数

        Returns:
            包含所有步骤结果的综合字典
        """
        all_results = {}

        if progress_callback:
            progress_callback("[1/7] 加载数据并初始化...")
        self.load_data(df)
        is_valid, msg = self.validate_target(target_column, task_type)
        if not is_valid:
            return {'error': msg}
        prep_res = self.prepare_datasets(sample_size=sample_size)
        all_results['prepare'] = prep_res

        if progress_callback:
            progress_callback("[2/7] 自动特征工程...")
        fe_res = self.run_feature_engineering(
            text_strategy=text_strategy,
            enable_poly_cross=enable_poly_cross,
            corr_threshold=corr_threshold,
            max_tfidf_features=max_tfidf_features,
            n_bins=n_bins,
        )
        all_results['feature_engineering'] = fe_res

        if progress_callback:
            progress_callback("[3/7] 特征重要性评估与选择...")
        fs_res = self.run_feature_selection(
            n_methods_required=n_methods_required,
            n_estimators=n_estimators,
            n_features=n_features,
            auto=auto,
            auto_threshold=auto_threshold,
        )
        all_results['feature_selection'] = fs_res

        if progress_callback:
            progress_callback("[4/7] 自动模型选择...")
        ms_res = self.run_model_selection(
            n_trials=n_trials,
            cv=cv,
            progress_callback=progress_callback,
        )
        all_results['model_selection'] = ms_res

        if progress_callback:
            progress_callback("[5/7] 模型诊断...")
        diag_res = self.run_diagnosis()
        all_results['diagnosis'] = diag_res

        if progress_callback:
            progress_callback("[6/7] 模型可解释性分析...")
        try:
            interp_res = self.run_interpretability(output_dir=interpretability_output_dir)
            all_results['interpretability'] = interp_res
        except Exception as e:
            all_results['interpretability'] = {'error': str(e)}

        if enable_drift_detection:
            if progress_callback:
                progress_callback("[6.5/7] 数据漂移检测...")
            try:
                ref_data = drift_reference_data
                new_data = None
                dataset_name = 'training_baseline'

                if ref_data is not None:
                    if self.diagnostician and hasattr(self.diagnostician, 'X_test') and self.diagnostician.X_test is not None:
                        new_data = self.diagnostician.X_test
                else:
                    if self.diagnostician and hasattr(self.diagnostician, 'X_train') and self.diagnostician.X_train is not None:
                        ref_data = self.diagnostician.X_train
                        if hasattr(self.diagnostician, 'X_test') and self.diagnostician.X_test is not None:
                            new_data = self.diagnostician.X_test

                if ref_data is not None:
                    drift_res = self.run_drift_detection(
                        reference_data=ref_data,
                        new_data=new_data,
                        storage_path=drift_storage_path,
                        dataset_name=dataset_name,
                        save_alert=True,
                    )
                    all_results['drift_detection'] = drift_res
            except Exception as e:
                all_results['drift_detection'] = {'error': str(e)}

        all_results['success'] = True
        all_results['best_model_name'] = ms_res.get('best_model_name', '')
        all_results['best_score'] = ms_res.get('best_score', 0.0)

        return all_results
