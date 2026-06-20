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

        learning_curve_data = self.diagnostician.learning_curve_analysis(
            best_model, X, y, cv=3
        )

        return {
            'metrics': metrics,
            'shap_data': shap_data,
            'is_overfitting': metrics.get('is_overfitting', False),
            'overfitting_suggestions': self.diagnostician.get_overfitting_suggestions() if metrics.get('is_overfitting', False) else [],
            'learning_curve': learning_curve_data,
        }

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
