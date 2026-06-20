"""
测试可解释性模块的4个增强功能：
1. 局部解释面板样本对比模式
2. 全局解释面板PDP多模型叠加
3. 对抗性检测特征归因路径追踪
4. 报告导出模型间解释一致性对比
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
print('测试可解释性模块增强功能')
print(sep)

# 创建测试数据
np.random.seed(42)
X, y = make_classification(
    n_samples=300, n_features=10, n_informative=6, n_redundant=2,
    n_clusters_per_class=2, random_state=42
)
feature_names = [f'feat_{i}' for i in range(10)]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42
)

column_types = {name: 'numeric' for name in feature_names}
print(f'测试数据: {X_df.shape[0]} 样本, {X_df.shape[1]} 特征')

# 训练多个模型用于测试
print('\n训练模型...')
models = {
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'  {name}: 准确率 = {score:.3f}')

best_model = models['RandomForest']
best_model_name = 'RandomForest'
all_models_list = [(name, model) for name, model in models.items()]

# 1. 测试局部解释样本对比模式
print('\n' + sep)
print('1. 测试局部解释样本对比模式')
print(sep)
try:
    from src.interpretability import LocalInterpreter

    local_interp = LocalInterpreter(
        best_model, X_df, 'binary', feature_names,
        random_state=42,
    )

    # 找一个正例和一个反例
    pos_idx = y_series[y_series == 1].index[0]
    neg_idx = y_series[y_series == 0].index[0]

    print(f'  对比样本: 正例 #{pos_idx} vs 反例 #{neg_idx}')

    compare_result = local_interp.explain_compare(pos_idx, neg_idx)

    if 'error' in compare_result:
        print(f'  [FAIL] {compare_result["error"]}')
    else:
        print(f'  样本1 (正例) 预测概率: {compare_result["sample1"]["prediction"]:.4f}')
        print(f'  样本2 (反例) 预测概率: {compare_result["sample2"]["prediction"]:.4f}')
        
        dd = compare_result["decision_difference"]
        if dd:
            print(f'  预测差异: {dd["prediction_diff"]:.4f}')
            print(f'  方向相反的特征数: {dd["n_opposite_features"]}')
        else:
            print(f'  预测差异: {abs(compare_result["sample1"]["prediction"] - compare_result["sample2"]["prediction"]):.4f}')
            print(f'  警告: 无法计算决策差异（SHAP explainer 不可用）')

        if compare_result["decision_difference"] and compare_result["decision_difference"]["opposite_features"]:
            df_opposite = pd.DataFrame(compare_result["decision_difference"]["opposite_features"])
            print(f'  TOP-5 决策差异特征:')
            for _, row in df_opposite.head(5).iterrows():
                print(f'    {row["feature"]}: shap1={row["sample1_shap"]:.4f}, shap2={row["sample2_shap"]:.4f}, '
                      f'diff={row["diff_abs"]:.4f}, direction={row["direction"]}')

        print(f'  样本1一致性评分: {compare_result["sample1"]["consistency_score"]:.3f}')
        print(f'  样本2一致性评分: {compare_result["sample2"]["consistency_score"]:.3f}')
        print('  [OK] 局部解释样本对比模式测试通过')
except Exception as e:
    print(f'  [FAIL] 局部解释样本对比模式测试失败: {e}')
    import traceback
    traceback.print_exc()

# 2. 测试全局解释多模型PDP
print('\n' + sep)
print('2. 测试全局解释多模型PDP')
print(sep)
try:
    from src.interpretability import GlobalInterpreter

    global_interp = GlobalInterpreter(
        best_model, X_df, y_series, 'binary', feature_names,
        random_state=42,
    )

    feature = 'feat_0'
    print(f'  测试特征: {feature}')
    print(f'  模型数量: {len(all_models_list)}')

    pdp_multi_result = global_interp.compute_pdp_multi(feature, all_models_list)

    if 'error' in pdp_multi_result:
        print(f'  [FAIL] {pdp_multi_result["error"]}')
    else:
        print(f'  特征: {pdp_multi_result["feature"]}')
        print(f'  模型结果数: {len(pdp_multi_result["model_results"])}')

        for mr in pdp_multi_result["model_results"]:
            print(f'    {mr["model_name"]}: grid点数={len(mr["grid_values"])}, '
                  f'PDP范围=[{mr["pdp_values"].min():.4f}, {mr["pdp_values"].max():.4f}]')

        # 测试单模型退化情况
        single_model_list = [('RandomForest', best_model)]
        pdp_single_result = global_interp.compute_pdp_multi(feature, single_model_list)

        if 'error' not in pdp_single_result:
            print(f'  单模型模式: {pdp_single_result["model_results"][0]["model_name"]}')
            print('  [OK] 全局解释多模型PDP测试通过')
except Exception as e:
    print(f'  [FAIL] 全局解释多模型PDP测试失败: {e}')
    import traceback
    traceback.print_exc()

# 3. 测试对抗性检测特征归因路径追踪
print('\n' + sep)
print('3. 测试对抗性检测特征归因路径追踪')
print(sep)
try:
    from src.interpretability import AdversarialExplainer

    adv = AdversarialExplainer(
        best_model, X_df, 'binary', feature_names, column_types,
        random_state=42,
    )

    # 先检测某个样本
    sample_idx = 5
    print(f'  检测样本 #{sample_idx}...')

    # 获取原始SHAP解释用于归因追踪
    local_interp = LocalInterpreter(
        best_model, X_df, 'binary', feature_names,
        random_state=42,
    )
    orig_shap = local_interp.explain_shap(sample_idx)

    # 追踪归因路径
    trace_result = adv.trace_attribution_path(sample_idx, orig_shap)

    if 'error' in trace_result:
        print(f'  [FAIL] {trace_result["error"]}')
    else:
        print(f'  样本 #{sample_idx} 结果:')
        print(f'    原始Kendall τ: {trace_result["original_tau"]:.4f}')
        print(f'    最小扰动后τ: {trace_result["min_tau"]:.4f}')
        print(f'    最大τ降幅: {trace_result["max_tau_drop"]:.4f}')
        print(f'    归因不稳定特征: {trace_result["trigger_feature"]}')
        print(f'    特征扰动数量: {len(trace_result["feature_tau_drops"])}')

        top_drops = sorted(
            trace_result["feature_tau_drops"].items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        print(f'    TOP-5 τ降幅特征:')
        for feat, drop in top_drops:
            print(f'      {feat}: {drop:.4f}')

    # 测试批量检测
    print(f'\n  批量检测 10 个样本...')
    batch_result = adv.batch_detect(n_samples=10)

    if 'error' in batch_result:
        print(f'  [FAIL] {batch_result["error"]}')
    else:
        print(f'    总样本数: {batch_result["total_samples"]}')
        print(f'    敏感样本数: {batch_result["n_sensitive"]}')
        print(f'    稳定样本数: {batch_result["n_stable"]}')
        print(f'    通过率: {batch_result["adversarial_pass_rate"]*100:.1f}%')

        if batch_result.get('sample_results'):
            has_trigger = sum(1 for r in batch_result['sample_results'] if r.get('trigger_feature'))
            print(f'    包含触发特征的样本数: {has_trigger}')

            for r in batch_result['sample_results'][:3]:
                print(f'      样本 #{r["sample_idx"]}: {r["label"]}, '
                      f'触发特征={r.get("trigger_feature", "-")}, '
                      f'Mean τ={r["mean_kendall_tau"]:.3f}')

        print('  [OK] 对抗性检测特征归因路径追踪测试通过')
except Exception as e:
    print(f'  [FAIL] 对抗性检测特征归因路径追踪测试失败: {e}')
    import traceback
    traceback.print_exc()

# 4. 测试报告导出模型间解释一致性对比
print('\n' + sep)
print('4. 测试报告导出模型间解释一致性对比')
print(sep)
try:
    from src.interpretability import InterpretabilityReportExporter

    reporter = InterpretabilityReportExporter(
        best_model, X_df, y_series, 'binary', best_model_name,
        feature_names, column_types, random_state=42,
        all_models=all_models_list,
    )

    # 测试模型一致性计算
    print(f'  计算模型间解释一致性...')
    print(f'  模型数量: {len(all_models_list)}')

    consistency_result = reporter.compute_model_consistency()

    if 'error' in consistency_result:
        print(f'  [FAIL] {consistency_result["error"]}')
    else:
        print(f'  一致性矩阵维度: {consistency_result["consistency_matrix"].shape}')
        print(f'  模型列表: {consistency_result["model_names"]}')

        df_matrix = pd.DataFrame(
            consistency_result["consistency_matrix"],
            index=consistency_result["model_names"],
            columns=consistency_result["model_names"],
        )
        print(f'  一致性矩阵:')
        print(df_matrix.round(3).to_string())

        print(f'  平均一致性: {consistency_result["avg_consistency"]:.4f}')
        print(f'  最低一致性对: {consistency_result["lowest_pair"][0]} - {consistency_result["lowest_pair"][1]} '
              f'(τ={consistency_result["lowest_tau"]:.4f})')
        print(f'  低一致性对数量 (τ<0.4): {len(consistency_result["low_consistency_pairs"])}')

        for pair in consistency_result["low_consistency_pairs"]:
            print(f'    解释分歧: {pair["model1"]} vs {pair["model2"]}, τ={pair["kendall_tau"]:.4f}')

        print('  [OK] 模型间解释一致性对比测试通过')

except Exception as e:
    print(f'  [FAIL] 模型间解释一致性对比测试失败: {e}')
    import traceback
    traceback.print_exc()

# 5. 测试完整HTML报告导出
print('\n' + sep)
print('5. 测试完整HTML报告导出')
print(sep)
try:
    reporter = InterpretabilityReportExporter(
        best_model, X_df.head(100), y_series.head(100), 'binary', best_model_name,
        feature_names, column_types, random_state=42,
        all_models=all_models_list,
    )

    output_dir = tempfile.mkdtemp()
    report_path = os.path.join(output_dir, 'interpretability_report.html')

    print(f'  报告输出路径: {report_path}')
    print(f'  这可能需要几分钟...')

    result = reporter.export_html_report(report_path)

    if 'error' in result:
        print(f'  [FAIL] {result["error"]}')
    else:
        print(f'  报告生成成功: {result.get("output_path")}')
        print(f'  文件存在: {os.path.exists(result.get("output_path", ""))}')

        summary = result.get('summary', {})
        print(f'  报告摘要:')
        print(f'    模型: {summary.get("model_name")}')
        print(f'    特征数: {summary.get("n_features")}')
        print(f'    样本数: {summary.get("n_samples")}')
        print(f'    整体一致性: {summary.get("overall_consistency", 0):.3f}')
        print(f'    可信度评级: {summary.get("credibility_grade")}')

        report_sections = summary.get('report_sections', [])
        print(f'  报告章节数: {len(report_sections)}')
        for section in report_sections:
            print(f'    ✓ {section}')

        print('  [OK] 完整HTML报告导出测试通过')
except Exception as e:
    print(f'  [FAIL] 完整HTML报告导出测试失败: {e}')
    import traceback
    traceback.print_exc()

# 6. 测试ModelSelector的get_top_models方法
print('\n' + sep)
print('6. 测试ModelSelector的get_top_models方法')
print(sep)
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
    print(f'  TOP-3 模型数量: {len(top_3)}')
    for i, (name, model, score) in enumerate(top_3, 1):
        print(f'    {i}. {name}: score={score:.3f}')

    top_1 = selector.get_top_models(n=1)
    print(f'  TOP-1 模型: {top_1[0][0]}')

    assert top_3[0][0] == 'RandomForest', 'TOP-1应该是RandomForest'
    assert len(top_3) == 3, '应该返回3个模型'

    print('  [OK] ModelSelector的get_top_models方法测试通过')
except Exception as e:
    print(f'  [FAIL] ModelSelector的get_top_models方法测试失败: {e}')
    import traceback
    traceback.print_exc()

print('\n' + sep)
print('所有可解释性增强功能测试完成！')
print(sep)
