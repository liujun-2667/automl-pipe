"""
测试新功能：数据质量评分、特征交互探索、学习曲线分析、漂移检测
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import tempfile

sep = '=' * 60
print(sep)
print('测试新功能模块')
print(sep)

# 1. 数据质量评分
print('\n1. 测试数据质量评分...')
try:
    from src.data_exploration import DataQualityScorer

    df = pd.DataFrame({
        '年龄': [25, 30, 35, np.nan, 45, 50, 55, 60, 65, 70] * 10,
        '收入': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 500000] * 10,
        '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'] * 10,
    })

    column_types = {'年龄': 'numeric', '收入': 'numeric', '性别': 'categorical'}

    scorer = DataQualityScorer(df, column_types)
    result = scorer.calculate_score()

    print(f'   综合评分: {result["overall_score"]}')
    print(f'   等级: {result["grade"]}')
    print(f'   缺失率评分: {result["missing"]["score"]} (缺失率: {result["missing"]["missing_rate"]}%)')
    print(f'   重复行评分: {result["duplicates"]["score"]} (重复率: {result["duplicates"]["duplicate_rate"]}%)')
    print(f'   异常值评分: {result["outliers"]["score"]} (高异常列占比: {result["outliers"]["outlier_cols_ratio"]}%)')
    print('   [OK] 数据质量评分测试通过')
except Exception as e:
    print(f'   [FAIL] 数据质量评分测试失败: {e}')
    import traceback
    traceback.print_exc()

# 2. 特征交互探索
print('\n2. 测试特征交互探索...')
try:
    from src.feature_engineering import FeatureInteractionExplorer

    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(20, 60, 200),
        'income': np.random.randint(30000, 100000, 200),
        'gender': np.random.choice(['男', '女'], 200),
        'city': np.random.choice(['北京', '上海', '深圳'], 200),
        'label': np.random.randint(0, 2, 200),
    })

    column_types = {
        'age': 'numeric',
        'income': 'numeric',
        'gender': 'categorical',
        'city': 'categorical',
        'label': 'numeric',
    }

    explorer = FeatureInteractionExplorer(df, column_types, 'label')

    # 数值-数值
    result1 = explorer.analyze_interaction('age', 'income', 'binary')
    print(f'   数值-数值交互: 类型={result1["type"]}, 相关系数={result1["correlation"]}, 互信息={result1["mi_info"]["mi_pair"]}')

    # 数值-分类
    result2 = explorer.analyze_interaction('age', 'gender', 'binary')
    print(f'   数值-分类交互: 类型={result2["type"]}, F统计量={result2["f_statistic"]}, p值={result2["p_value"]}')

    # 分类-分类
    result3 = explorer.analyze_interaction('gender', 'city', 'binary')
    print(f'   分类-分类交互: 类型={result3["type"]}, 卡方={result3["chi_square"]}, p值={result3["p_value"]}')

    print('   [OK] 特征交互探索测试通过')
except Exception as e:
    print(f'   [FAIL] 特征交互探索测试失败: {e}')
    import traceback
    traceback.print_exc()

# 3. 学习曲线分析
print('\n3. 测试学习曲线分析...')
try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from src.model_diagnosis import ModelDiagnostician

    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    y_series = pd.Series(y)

    diagnostician = ModelDiagnostician(task_type='binary', random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    result = diagnostician.learning_curve_analysis(model, X_df, y_series, cv=3)

    print(f'   训练比例: {result["train_sizes"]}')
    print(f'   训练得分: {result["train_scores_mean"]}')
    print(f'   验证得分: {result["test_scores_mean"]}')
    print(f'   是否过拟合: {result["is_overfitting"]}')
    print(f'   是否欠拟合: {result["is_underfitting"]}')
    print(f'   是否需要更多数据: {result["needs_more_data"]}')
    print(f'   建议数: {len(result["suggestions"])}')
    print('   [OK] 学习曲线分析测试通过')
except Exception as e:
    print(f'   [FAIL] 学习曲线分析测试失败: {e}')
    import traceback
    traceback.print_exc()

# 4. 漂移检测脚本生成
print('\n4. 测试漂移检测脚本生成...')
try:
    from src.pipeline_exporter import PipelineExporter
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    X_train = pd.DataFrame({
        'feat_0': np.random.normal(0, 1, 200),
        'feat_1': np.random.normal(5, 2, 200),
        'feat_2': np.random.choice(['a', 'b', 'c'], 200),
    })

    exporter = PipelineExporter(
        model=RandomForestClassifier(),
        column_types={'feat_0': 'numeric', 'feat_1': 'numeric', 'feat_2': 'categorical'},
        task_type='binary',
        feature_names=['feat_0', 'feat_1', 'feat_2'],
        target_column='target',
    )

    # 计算特征统计
    stats = exporter.compute_feature_stats(X_train)
    print(f'   统计计算: {len(stats["numeric"])} 数值列, {len(stats["categorical"])} 分类列')

    # 导出
    output_dir = tempfile.mkdtemp()
    results = exporter.export_all(output_dir, X_sample=X_train, include_onnx=False)

    print(f'   导出文件: {list(results.keys())}')
    print(f'   漂移脚本存在: {os.path.exists(results.get("drift_detector", ""))}')
    print(f'   统计文件存在: {os.path.exists(results.get("feature_stats", ""))}')

    # 测试漂移检测器
    import importlib.util
    spec = importlib.util.spec_from_file_location('drift_detector', results['drift_detector'])
    drift_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift_module)

    X_new = pd.DataFrame({
        'feat_0': np.random.normal(0, 1.1, 100),
        'feat_1': np.random.normal(8, 3, 100),
        'feat_2': np.random.choice(['a', 'b', 'c'], 100),
    })

    detector = drift_module.FeatureDriftDetector(results['feature_stats'])
    result = detector.detect_all_drift(X_new)

    print(f'   漂移检测: {result["summary"]["drifted_count"]}/{result["summary"]["total_features"]} 个特征漂移')
    print(f'   漂移特征: {result["summary"]["drifted_features"]}')

    print('   [OK] 漂移检测脚本测试通过')
except Exception as e:
    print(f'   [FAIL] 漂移检测脚本测试失败: {e}')
    import traceback
    traceback.print_exc()

# 5. 主Pipeline整合测试
print('\n5. 测试主Pipeline整合...')
try:
    from src.automl_pipeline import AutoMLPipeline

    pipeline = AutoMLPipeline(task_type='binary', target_column='label')

    np.random.seed(42)
    df = pd.DataFrame({
        '年龄': np.random.randint(20, 60, 200),
        '收入': np.random.randint(30000, 100000, 200),
        '性别': np.random.choice(['男', '女'], 200),
        'label': np.random.randint(0, 2, 200),
    })

    pipeline.load_data(df)
    pipeline.prepare_datasets()
    feat_result = pipeline.run_feature_engineering(enable_poly_cross=False)
    sel_result = pipeline.run_feature_selection(n_estimators=50)
    model_result = pipeline.run_model_selection(n_trials=3, cv=3)
    diag_result = pipeline.run_diagnosis()

    print(f'   诊断结果包含学习曲线: {"learning_curve" in diag_result}')
    if 'learning_curve' in diag_result:
        lc = diag_result['learning_curve']
        print(f'   学习曲线点数: {len(lc["train_sizes"])}')
        print(f'   最终训练得分: {lc["final_train_score"]}')
        print(f'   最终验证得分: {lc["final_test_score"]}')

    print('   [OK] 主Pipeline整合测试通过')
except Exception as e:
    print(f'   [FAIL] 主Pipeline整合测试失败: {e}')
    import traceback
    traceback.print_exc()

print('\n' + sep)
print('所有新功能测试完成！')
print(sep)
