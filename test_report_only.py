"""
单独测试报告导出功能
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sep = '=' * 70
print(sep)
print('单独测试报告导出功能')
print(sep)

np.random.seed(42)
X, y = make_classification(
    n_samples=100, n_features=10, n_informative=6, n_redundant=2,
    n_clusters_per_class=2, random_state=42
)
feature_names = [f'feat_{i}' for i in range(10)]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42
)

column_types = {name: 'numeric' for name in feature_names}
print(f'测试数据: {X_df.shape[0]} 样本, {X_df.shape[1]} 特征', flush=True)

print('\n训练模型...', flush=True)
models = {
    'RandomForest': RandomForestClassifier(n_estimators=30, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=30, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'  {name}: 准确率 = {score:.3f}', flush=True)

best_model = models['RandomForest']
best_model_name = 'RandomForest'
all_models_list = [(name, model) for name, model in models.items()]

# 测试报告导出模型间一致性
print('\n' + sep)
print('4. 测试报告导出模型间解释一致性对比')
print(sep, flush=True)
try:
    from src.interpretability import InterpretabilityReportExporter

    reporter = InterpretabilityReportExporter(
        best_model, X_df, y_series, 'binary', best_model_name,
        feature_names, column_types, random_state=42,
        all_models=all_models_list,
    )

    print(f'  计算模型间解释一致性...', flush=True)
    print(f'  模型数量: {len(all_models_list)}', flush=True)

    consistency_result = reporter.compute_model_consistency()

    if 'error' in consistency_result:
        print(f'  [FAIL] {consistency_result["error"]}', flush=True)
    else:
        print(f'  一致性矩阵维度: {consistency_result["consistency_matrix"].shape}', flush=True)
        print(f'  模型列表: {consistency_result["model_names"]}', flush=True)

        df_matrix = pd.DataFrame(
            consistency_result["consistency_matrix"],
            index=consistency_result["model_names"],
            columns=consistency_result["model_names"],
        )
        print(f'  一致性矩阵:', flush=True)
        print(df_matrix.round(3).to_string(), flush=True)

        print(f'  平均一致性: {consistency_result["avg_consistency"]:.4f}', flush=True)
        print(f'  最低一致性对: {consistency_result["lowest_pair"][0]} - {consistency_result["lowest_pair"][1]} '
              f'(τ={consistency_result["lowest_tau"]:.4f})', flush=True)
        print(f'  低一致性对数量 (τ<0.4): {len(consistency_result["low_consistency_pairs"])}', flush=True)

        for pair in consistency_result["low_consistency_pairs"]:
            print(f'    解释分歧: {pair["model1"]} vs {pair["model2"]}, τ={pair["kendall_tau"]:.4f}', flush=True)

        print('  [OK] 模型间解释一致性对比测试通过', flush=True)

except Exception as e:
    print(f'  [FAIL] 模型间解释一致性对比测试失败: {e}', flush=True)
    import traceback
    traceback.print_exc()

# 6. 测试ModelSelector的get_top_models方法
print('\n' + sep)
print('6. 测试ModelSelector的get_top_models方法')
print(sep, flush=True)
try:
    from src.model_selector import AutoModelSelector

    selector = AutoModelSelector(task_type='binary', random_state=42)
    selector.models_ = models
    selector.results_ = [
        {'model_name': 'RandomForest', 'cv_mean': 0.92},
        {'model_name': 'GradientBoosting', 'cv_mean': 0.88},
        {'model_name': 'LogisticRegression', 'cv_mean': 0.82},
    ]

    top_3 = selector.get_top_models(n=3)
    print(f'  TOP-3 模型数量: {len(top_3)}', flush=True)
    for i, (name, model, score) in enumerate(top_3, 1):
        print(f'    {i}. {name}: score={score:.3f}', flush=True)

    top_1 = selector.get_top_models(n=1)
    print(f'  TOP-1 模型: {top_1[0][0]}', flush=True)

    assert top_3[0][0] == 'RandomForest', 'TOP-1应该是RandomForest'
    assert len(top_3) == 3, '应该返回3个模型'

    print('  [OK] ModelSelector的get_top_models方法测试通过', flush=True)
except Exception as e:
    print(f'  [FAIL] ModelSelector的get_top_models方法测试失败: {e}', flush=True)
    import traceback
    traceback.print_exc()

print('\n' + sep)
print('测试完成！')
print(sep, flush=True)
