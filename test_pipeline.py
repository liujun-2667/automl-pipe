"""
测试脚本 - 验证AutoML Pipeline各模块功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

sep = '=' * 60
print(sep)
print('开始测试 AutoML Pipeline 各模块')
print(sep)

# 1. 数据探索模块
print('\n1. 测试数据探索模块...')
try:
    from src.data_exploration import DataTypeInference, DataExplorer, DataSampler, TargetValidator
    
    df = pd.DataFrame({
        '年龄': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        '收入': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
        '是否购买': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    })
    
    types = DataTypeInference.infer_all(df)
    print('   数据类型推断:', types)
    
    explorer = DataExplorer(df, types)
    overview = explorer.get_overview()
    print('   数据概况:', overview)
    
    is_valid, msg = TargetValidator.validate(df, '是否购买', 'binary')
    print('   目标列校验:', is_valid, msg)
    
    print('   [OK] 数据探索模块测试通过')
except Exception as e:
    print('   [FAIL] 数据探索模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 2. 特征工程模块
print('\n2. 测试特征工程模块...')
try:
    from src.feature_engineering import AutoFeatureEngineer
    
    df = pd.DataFrame({
        '年龄': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        '收入': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
    })
    y = pd.Series([0, 1, 0, 1, 1, 1, 0, 1, 0, 1], name='label')
    
    column_types = {
        '年龄': 'numeric',
        '收入': 'numeric',
        '性别': 'categorical',
    }
    
    fe = AutoFeatureEngineer(column_types=column_types, task_type='binary')
    result = fe.fit_transform(df, y)
    print('   特征工程完成，生成', result.shape[1], '个特征')
    print('   特征名示例:', list(result.columns[:5]))
    
    print('   [OK] 特征工程模块测试通过')
except Exception as e:
    print('   [FAIL] 特征工程模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 3. 特征选择模块
print('\n3. 测试特征选择模块...')
try:
    from sklearn.datasets import make_classification
    from src.feature_selection import FeatureImportanceAnalyzer, IntersectionFeatureSelector
    
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, n_redundant=5,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=['feat_' + str(i) for i in range(20)])
    y_series = pd.Series(y)
    
    analyzer = FeatureImportanceAnalyzer(task_type='binary', n_estimators=50)
    analyzer.fit(X_df, y_series)
    print('   特征重要性分析完成')
    
    selector = IntersectionFeatureSelector(n_methods_required=2)
    selector.fit(analyzer, n_features=10, auto=False)
    print('   选中', len(selector.selected_features_), '个特征')
    
    print('   [OK] 特征选择模块测试通过')
except Exception as e:
    print('   [FAIL] 特征选择模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 4. 模型选择模块
print('\n4. 测试模型选择模块...')
try:
    from sklearn.datasets import make_classification
    from src.model_selector import AutoModelSelector
    
    selector = AutoModelSelector(task_type='binary', n_trials=5, cv=3)
    
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    X_df = pd.DataFrame(X, columns=['feat_' + str(i) for i in range(10)])
    y_series = pd.Series(y)
    
    models = selector.get_model_definitions()
    print('   模型定义:', [m.name for m in models])
    
    results = selector.fit(X_df, y_series, X_df, y_series)
    print('   模型选择完成，最佳模型:', selector.get_best_model_name())
    print('   最佳得分:', round(selector.best_score_, 4))
    
    print('   [OK] 模型选择模块测试通过')
except Exception as e:
    print('   [FAIL] 模型选择模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 5. 模型诊断模块
print('\n5. 测试模型诊断模块...')
try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from src.model_diagnosis import ModelDiagnostician
    
    diagnostician = ModelDiagnostician(task_type='binary')
    
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    X_df = pd.DataFrame(X, columns=['feat_' + str(i) for i in range(10)])
    y_series = pd.Series(y)
    
    diagnostician.split_data(X_df, y_series)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    metrics = diagnostician.evaluate_model(model)
    
    print('   诊断完成，准确率:', round(metrics.get('accuracy', 0), 4))
    print('   是否过拟合:', metrics.get('is_overfitting', False))
    
    print('   [OK] 模型诊断模块测试通过')
except Exception as e:
    print('   [FAIL] 模型诊断模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 6. Pipeline导出模块
print('\n6. 测试Pipeline导出模块...')
try:
    from sklearn.ensemble import RandomForestClassifier
    from src.pipeline_exporter import PipelineExporter
    import tempfile
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    exporter = PipelineExporter(
        model=model,
        column_types={'feat1': 'numeric', 'feat2': 'categorical'},
        task_type='binary',
        feature_names=['feat1', 'feat2'],
        target_column='target',
    )
    
    exporter.set_model_info(
        model_name='随机森林',
        best_params={'n_estimators': 50},
        cv_mean=0.85,
        cv_std=0.03,
        train_time=10.5,
    )
    
    exporter.set_dataset_info(
        n_rows=1000,
        n_cols=10,
        n_features=20,
        feature_list=['feat1', 'feat2'],
    )
    
    output_dir = tempfile.mkdtemp()
    
    result = exporter.export_all(output_dir, include_onnx=False)
    
    print('   导出文件:', list(result.keys()))
    for key, path in result.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print('     -', key, ':', path, '(', size, 'bytes )')
    
    print('   [OK] Pipeline导出模块测试通过')
except Exception as e:
    print('   [FAIL] Pipeline导出模块测试失败:', str(e))
    import traceback
    traceback.print_exc()

# 7. 主Pipeline整合测试
print('\n7. 测试主Pipeline整合...')
try:
    from src.automl_pipeline import AutoMLPipeline
    
    pipeline = AutoMLPipeline(task_type='binary', target_column='label')
    
    df = pd.DataFrame({
        '年龄': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
        '收入': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000] * 10,
        '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'] * 10,
        'label': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1] * 10,
    })
    
    pipeline.load_data(df)
    print('   数据加载完成，列数:', len(pipeline.column_types))
    
    pipeline.prepare_datasets()
    print('   数据集准备完成')
    
    feat_result = pipeline.run_feature_engineering(enable_poly_cross=False)
    print('   特征工程完成，特征数:', feat_result['n_features_full'])
    
    sel_result = pipeline.run_feature_selection(n_estimators=50)
    print('   特征选择完成，选中:', sel_result['n_selected'], '个特征')
    
    model_result = pipeline.run_model_selection(n_trials=3, cv=3)
    print('   模型选择完成，最佳:', model_result['best_model_name'])
    
    diag_result = pipeline.run_diagnosis()
    print('   模型诊断完成')
    
    print('   [OK] 主Pipeline整合测试通过')
except Exception as e:
    print('   [FAIL] 主Pipeline整合测试失败:', str(e))
    import traceback
    traceback.print_exc()

print('\n' + sep)
print('测试完成！')
print(sep)
