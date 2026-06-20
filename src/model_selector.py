"""
自动模型选择与超参搜索模块
- 6种候选模型
- 贝叶斯优化(Optuna)超参搜索
- 交叉验证评估
- 进度追踪
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, f1_score, mean_squared_error, r2_score,
    accuracy_score, make_scorer
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import optuna
import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelDefinition:
    """模型定义"""

    def __init__(self, name: str, model_class, param_space_func: Callable):
        self.name = name
        self.model_class = model_class
        self.param_space_func = param_space_func


class AutoModelSelector:
    """自动模型选择器"""

    def __init__(
        self,
        task_type: str = 'binary',
        n_trials: int = 30,
        cv: int = 5,
        random_state: int = 42,
        timeout: Optional[int] = None,
    ):
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.timeout = timeout

        self.models_ = {}
        self.results_ = []
        self.best_model_name_ = None
        self.best_model_ = None
        self.best_params_ = {}
        self.best_score_ = -np.inf

        self.trial_progress = {
            'current_model': '',
            'current_trial': 0,
            'total_trials': 0,
            'current_best_score': 0.0,
            'is_running': False,
        }

        self._stop_flag = False
        self._trial_callbacks = []

    def get_model_definitions(self) -> List[ModelDefinition]:
        """获取所有候选模型定义"""
        if self.task_type in ['binary', 'multiclass']:
            return self._get_classification_models()
        else:
            return self._get_regression_models()

    def _get_classification_models(self) -> List[ModelDefinition]:
        """分类模型定义"""

        def lr_params(trial: optuna.Trial) -> Dict:
            return {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': self.random_state,
            }

        def rf_params(trial: optuna.Trial) -> Dict:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': self.random_state,
                'n_jobs': -1,
            }

        def svm_params(trial: optuna.Trial) -> Dict:
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'probability': True,
                'random_state': self.random_state,
                'max_iter': 2000,
            }

        def knn_params(trial: optuna.Trial) -> Dict:
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 3),
                'n_jobs': -1,
            }

        def mlp_params(trial: optuna.Trial) -> Dict:
            hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
            hidden_units = []
            for i in range(hidden_layers):
                hidden_units.append(trial.suggest_int(f'hidden_unit_{i}', 32, 256))
            return {
                'hidden_layer_sizes': tuple(hidden_units),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'early_stopping': True,
                'random_state': self.random_state,
            }

        def xgb_params(trial: optuna.Trial) -> Dict:
            try:
                from xgboost import XGBClassifier
            except ImportError:
                return rf_params(trial)

            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
                'random_state': self.random_state,
                'use_label_encoder': False,
                'verbosity': 0,
            }

        models = [
            ModelDefinition('逻辑回归', LogisticRegression, lr_params),
            ModelDefinition('随机森林', RandomForestClassifier, rf_params),
            ModelDefinition('XGBoost', self._get_xgb_classifier, xgb_params),
            ModelDefinition('SVM', SVC, svm_params),
            ModelDefinition('K近邻', KNeighborsClassifier, knn_params),
            ModelDefinition('MLP', MLPClassifier, mlp_params),
        ]

        return models

    def _get_regression_models(self) -> List[ModelDefinition]:
        """回归模型定义"""

        def lr_params(trial: optuna.Trial) -> Dict:
            return {
                'fit_intercept': True,
            }

        def rf_params(trial: optuna.Trial) -> Dict:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': self.random_state,
                'n_jobs': -1,
            }

        def svm_params(trial: optuna.Trial) -> Dict:
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'max_iter': 2000,
            }

        def knn_params(trial: optuna.Trial) -> Dict:
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 3),
                'n_jobs': -1,
            }

        def mlp_params(trial: optuna.Trial) -> Dict:
            hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
            hidden_units = []
            for i in range(hidden_layers):
                hidden_units.append(trial.suggest_int(f'hidden_unit_{i}', 32, 256))
            return {
                'hidden_layer_sizes': tuple(hidden_units),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'early_stopping': True,
                'random_state': self.random_state,
            }

        def xgb_params(trial: optuna.Trial) -> Dict:
            try:
                from xgboost import XGBRegressor
            except ImportError:
                return rf_params(trial)

            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
                'random_state': self.random_state,
                'verbosity': 0,
            }

        models = [
            ModelDefinition('线性回归', LinearRegression, lr_params),
            ModelDefinition('随机森林', RandomForestRegressor, rf_params),
            ModelDefinition('XGBoost', self._get_xgb_regressor, xgb_params),
            ModelDefinition('SVR', SVR, svm_params),
            ModelDefinition('K近邻', KNeighborsRegressor, knn_params),
            ModelDefinition('MLP', MLPRegressor, mlp_params),
        ]

        return models

    def _get_xgb_classifier(self, **kwargs):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(**kwargs)
        except ImportError:
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            )

    def _get_xgb_regressor(self, **kwargs):
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**kwargs)
        except ImportError:
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            )

    def _get_scoring(self) -> str:
        """获取评估指标"""
        if self.task_type == 'binary':
            return 'roc_auc'
        elif self.task_type == 'multiclass':
            return 'f1_macro'
        else:
            return 'neg_root_mean_squared_error'

    def _get_cv_splitter(self):
        """获取交叉验证分割器"""
        if self.task_type in ['binary', 'multiclass']:
            return StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )

    def optimize_model(
        self,
        model_def: ModelDefinition,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict:
        """对单个模型进行贝叶斯优化"""
        self.trial_progress['current_model'] = model_def.name
        self.trial_progress['current_trial'] = 0
        self.trial_progress['total_trials'] = self.n_trials

        best_model_params = {}
        best_trial_score = -np.inf

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_model_params, best_trial_score

            if self._stop_flag:
                raise optuna.TrialPruned()

            params = model_def.param_space_func(trial)

            if callable(model_def.model_class) and not isinstance(model_def.model_class, type):
                model = model_def.model_class(**params)
            else:
                model = model_def.model_class(**params)

            cv = self._get_cv_splitter()
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=self._get_scoring(),
                    n_jobs=-1,
                    error_score='raise',
                )
                score = np.mean(scores)
            except Exception as e:
                score = -np.inf

            if score > best_trial_score:
                best_trial_score = score
                best_model_params = params

            self.trial_progress['current_trial'] = trial.number + 1
            if score > self.trial_progress['current_best_score']:
                self.trial_progress['current_best_score'] = score

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )
        except optuna.TrialPruned:
            pass

        if best_trial_score == -np.inf:
            best_params = {}
            best_score = -np.inf
        else:
            best_params = best_model_params
            best_score = best_trial_score

        return {
            'model_name': model_def.name,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials_completed': len(study.trials),
        }

    def final_evaluation(
        self,
        model_def: ModelDefinition,
        best_params: Dict,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict:
        """在完整数据上做5折交叉验证最终评估"""
        if callable(model_def.model_class) and not isinstance(model_def.model_class, type):
            model = model_def.model_class(**best_params)
        else:
            model = model_def.model_class(**best_params)

        cv = self._get_cv_splitter()

        start_time = time.time()
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=self._get_scoring(),
            n_jobs=-1,
        )
        train_time = time.time() - start_time

        return {
            'model_name': model_def.name,
            'best_params': best_params,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_scores': scores.tolist(),
            'train_time': train_time,
        }

    def fit(
        self,
        X_fast: pd.DataFrame,
        y_fast: pd.Series,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        执行完整的模型选择流程

        Args:
            X_fast: 快速实验子集特征
            y_fast: 快速实验子集目标
            X_full: 完整数据集特征
            y_full: 完整数据集目标
            progress_callback: 进度回调函数
        """
        self._stop_flag = False
        self.trial_progress['is_running'] = True
        self.results_ = []

        model_defs = self.get_model_definitions()

        for i, model_def in enumerate(model_defs):
            if self._stop_flag:
                break

            if progress_callback:
                progress_callback(f"正在优化模型: {model_def.name} ({i+1}/{len(model_defs)})")

            opt_result = self.optimize_model(model_def, X_fast, y_fast)

            if opt_result['best_score'] == -np.inf:
                continue

            if progress_callback:
                progress_callback(f"正在评估模型: {model_def.name} (完整数据CV)")

            eval_result = self.final_evaluation(
                model_def,
                opt_result['best_params'],
                X_full,
                y_full,
            )
            self.results_.append(eval_result)

            if callable(model_def.model_class) and not isinstance(model_def.model_class, type):
                best_model = model_def.model_class(**opt_result['best_params'])
            else:
                best_model = model_def.model_class(**opt_result['best_params'])
            best_model.fit(X_full, y_full)
            self.models_[model_def.name] = best_model

            if eval_result['cv_mean'] > self.best_score_:
                self.best_score_ = eval_result['cv_mean']
                self.best_model_name_ = model_def.name
                self.best_model_ = best_model
                self.best_params_ = opt_result['best_params']

        self.trial_progress['is_running'] = False
        return self.get_results_df()

    def stop(self):
        """停止搜索"""
        self._stop_flag = True

    def get_results_df(self) -> pd.DataFrame:
        """获取结果DataFrame"""
        if not self.results_:
            return pd.DataFrame()

        df = pd.DataFrame(self.results_)
        df = df.sort_values('cv_mean', ascending=False).reset_index(drop=True)

        if self.task_type == 'regression':
            df['cv_mean'] = -df['cv_mean']
            df['cv_std'] = df['cv_std'].abs()

        return df

    def get_best_score(self) -> float:
        """获取最佳模型得分（对外展示：回归为正RMSE）"""
        if self.task_type == 'regression':
            return -self.best_score_
        return self.best_score_

    def get_best_model(self):
        """获取最佳模型"""
        return self.best_model_

    def get_best_model_name(self) -> str:
        """获取最佳模型名称"""
        return self.best_model_name_

    def get_progress(self) -> Dict:
        """获取当前进度"""
        progress = self.trial_progress.copy()
        if self.task_type == 'regression' and progress.get('current_best_score', 0) != 0:
            progress['current_best_score'] = -progress['current_best_score']
        return progress

    def get_top_models(self, n: int = 3) -> List[Tuple[str, Any, float]]:
        """获取TOP-N模型列表

        Args:
            n: 要获取的模型数量

        Returns:
            [(model_name, model_instance, cv_score), ...]，按得分降序排列
        """
        if not self.results_:
            return []

        df = self.get_results_df()
        top_results = df.head(n)

        top_models = []
        for _, row in top_results.iterrows():
            model_name = row['model_name']
            if model_name in self.models_:
                score = row['cv_mean']
                top_models.append((model_name, self.models_[model_name], float(score)))

        return top_models
