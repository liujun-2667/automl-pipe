"""
AutoML Pipeline - Streamlit 主应用
表格数据自动特征工程与模型选择Pipeline工具
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from src.automl_pipeline import AutoMLPipeline
from src.data_exploration import DataTypeInference, DataExplorer
from src.interpretability import (
    LocalInterpreter, GlobalInterpreter,
    AdversarialExplainer, InterpretabilityReportExporter,
    ModelInterpretabilityAnalyzer,
)
from src.drift_detection import (
    DriftDetector, AlertStorage, DriftReportExporter,
    compute_weighted_psi, SlidingWindowDriftMonitor, DriftTrendTracker,
)

st.set_page_config(
    page_title="AutoML Pipeline - 自动特征工程与模型选择",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def init_session_state():
    """初始化会话状态"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = AutoMLPipeline()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
    if 'target_col' not in st.session_state:
        st.session_state.target_col = ''
    if 'task_type' not in st.session_state:
        st.session_state.task_type = 'binary'
    if 'feature_engineering_result' not in st.session_state:
        st.session_state.feature_engineering_result = None
    if 'feature_selection_result' not in st.session_state:
        st.session_state.feature_selection_result = None
    if 'model_selection_result' not in st.session_state:
        st.session_state.model_selection_result = None
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None
    if 'interpretability_result' not in st.session_state:
        st.session_state.interpretability_result = None
    if 'drift_detection_result' not in st.session_state:
        st.session_state.drift_detection_result = None
    if 'drift_reference_df' not in st.session_state:
        st.session_state.drift_reference_df = None
    if 'drift_new_df' not in st.session_state:
        st.session_state.drift_new_df = None
    if 'drift_detection_mode' not in st.session_state:
        st.session_state.drift_detection_mode = 'once'
    if 'drift_monitor' not in st.session_state:
        st.session_state.drift_monitor = None
    if 'drift_trend_tracker' not in st.session_state:
        st.session_state.drift_trend_tracker = DriftTrendTracker()
    if 'drift_auto_monitoring' not in st.session_state:
        st.session_state.drift_auto_monitoring = False
    if 'drift_auto_interval' not in st.session_state:
        st.session_state.drift_auto_interval = 30
    if 'drift_window_result' not in st.session_state:
        st.session_state.drift_window_result = None
    if 'drift_weighted_psi' not in st.session_state:
        st.session_state.drift_weighted_psi = None
    if 'drift_feature_weights' not in st.session_state:
        st.session_state.drift_feature_weights = None
    if 'drift_last_detector' not in st.session_state:
        st.session_state.drift_last_detector = None
    if 'drift_last_ref_df' not in st.session_state:
        st.session_state.drift_last_ref_df = None
    if 'drift_last_new_df' not in st.session_state:
        st.session_state.drift_last_new_df = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False


def step_navigation():
    """步骤导航侧边栏"""
    st.sidebar.title("🔬 AutoML Pipeline")
    st.sidebar.markdown("---")

    steps = [
        "[1] 数据上传与探索",
        "[2] 自动特征工程",
        "[3] 特征重要性评估",
        "[4] 自动模型选择",
        "[5] 模型对比与诊断",
        "[6] 模型可解释性分析",
        "[7] 数据漂移检测",
        "[8] Pipeline导出",
    ]

    for i, step in enumerate(steps):
        if i == st.session_state.current_step:
            st.sidebar.markdown(f"**▶ {step}**")
        elif i < st.session_state.current_step:
            st.sidebar.markdown(f"✓ {step}")
        else:
            st.sidebar.markdown(f"○ {step}")

    st.sidebar.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("<- 上一步", disabled=st.session_state.current_step == 0):
            st.session_state.current_step = max(0, st.session_state.current_step - 1)
            st.rerun()
    with col2:
        if st.button("下一步 ->", disabled=st.session_state.current_step >= 7):
            st.session_state.current_step = min(7, st.session_state.current_step + 1)
            st.rerun()


def step_data_upload():
    """步骤1: 数据上传与探索"""
    st.title("📁 数据上传与探索")
    st.markdown("上传CSV格式的结构化数据集，系统将自动进行数据类型推断和概况分析。")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择CSV文件（最大500MB）",
            type=['csv'],
            help="支持中文列名，文件大小不超过500MB",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df

                if not st.session_state.column_types:
                    st.session_state.column_types = DataTypeInference.infer_all(df)

                st.success(f"✅ 成功加载数据，共 {len(df)} 行，{len(df.columns)} 列")

            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")
                return
        else:
            st.info("👆 请上传CSV文件开始")
            return

    with col2:
        st.metric("数据行数", f"{len(st.session_state.df):,}")
        st.metric("数据列数", len(st.session_state.df.columns))
        st.metric("内存占用", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    st.markdown("---")

    st.subheader("📋 数据预览")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("🏷️ 数据类型推断")
    st.caption("系统自动推断每列的数据类型，您可以手动修正")

    type_options = ['numeric', 'categorical', 'date', 'text', 'id']
    type_cn = {
        'numeric': '数值型',
        'categorical': '分类型',
        'date': '日期型',
        'text': '文本型',
        'id': 'ID型',
    }

    col_types = st.columns(3)
    for i, (col, col_type) in enumerate(st.session_state.column_types.items()):
        with col_types[i % 3]:
            new_type = st.selectbox(
                f"{col}",
                type_options,
                index=type_options.index(col_type) if col_type in type_options else 0,
                format_func=lambda x: type_cn[x],
                key=f"type_{col}",
            )
            if new_type != col_type:
                st.session_state.column_types[col] = new_type
                st.session_state.pipeline.update_column_type(col, new_type)

    st.markdown("---")

    st.subheader("📊 数据质量评分")

    from src.data_exploration import DataQualityScorer

    scorer = DataQualityScorer(st.session_state.df, st.session_state.column_types)
    quality_score = scorer.calculate_score()

    grade_colors = {
        '优秀': 'green',
        '良好': 'blue',
        '一般': 'orange',
        '较差': 'red',
    }
    grade_color = grade_colors.get(quality_score['grade'], 'gray')

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric(
            "综合评分",
            f"{quality_score['overall_score']} 分",
            f"等级: {quality_score['grade']}",
        )
        st.markdown(f"<h4 style='color: {grade_color}; text-align: center;'>数据质量：{quality_score['grade']}</h4>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 1.5))
        score = quality_score['overall_score']
        ax.barh([0], [score], color=grade_color, height=0.5)
        ax.barh([0], [100 - score], left=[score], color='#eee', height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 60, 75, 90, 100])
        ax.set_xticklabels(['0', '60', '75', '90', '100'])
        ax.axvline(x=60, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=75, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=90, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(score, 0, f' {score}', va='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        missing = quality_score['missing']
        st.metric("缺失率评分", f"{missing['score']} 分")
        st.caption(f"缺失率: {missing['missing_rate']}%")
        st.caption(f"缺失单元格: {missing['missing_cells']:,}")

    with col3:
        duplicates = quality_score['duplicates']
        st.metric("重复行评分", f"{duplicates['score']} 分")
        st.caption(f"重复率: {duplicates['duplicate_rate']}%")
        st.caption(f"重复行数: {duplicates['n_duplicates']:,}")

    with col4:
        outliers = quality_score['outliers']
        st.metric("异常值评分", f"{outliers['score']} 分")
        st.caption(f"高异常值列占比: {outliers['outlier_cols_ratio']}%")
        st.caption(f"异常值列数: {outliers['n_outlier_cols']}/{outliers['n_numeric_cols']}")

    with st.expander("📈 查看各列异常值详情"):
        if outliers['per_col_outliers']:
            outlier_df = pd.DataFrame([
                {'列名': col, '异常值比例(%)': ratio}
                for col, ratio in outliers['per_col_outliers'].items()
            ]).sort_values('异常值比例(%)', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.info("无数值列可分析")

    st.markdown("---")

    st.subheader("📋 数据概况")

    explorer = DataExplorer(st.session_state.df, st.session_state.column_types)
    overview = explorer.get_overview()
    all_stats = explorer.get_all_column_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("数值列", overview['n_numeric'])
    col2.metric("分类列", overview['n_categorical'])
    col3.metric("日期列", overview['n_date'])
    col4.metric("文本列", overview['n_text'])
    col5.metric("ID列", overview['n_id'])

    with st.expander("📈 查看详细统计信息"):
        stats_df = pd.DataFrame(all_stats)
        st.dataframe(stats_df, use_container_width=True)

    numeric_cols = [c for c, t in st.session_state.column_types.items() if t == 'numeric']
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.subheader("🔥 Pearson相关矩阵热力图")

        corr_matrix = explorer.get_correlation_matrix()
        if not corr_matrix.empty:
            fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix.columns) * 0.8), 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                ax=ax,
                annot_kws={'size': 8}
            )
            plt.title("特征相关矩阵热力图", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    st.subheader("🎯 目标列与任务类型")

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "选择目标列",
            options=st.session_state.df.columns.tolist(),
            index=st.session_state.df.columns.tolist().index(st.session_state.target_col)
            if st.session_state.target_col in st.session_state.df.columns else 0,
        )
        st.session_state.target_col = target_col

    with col2:
        task_type = st.selectbox(
            "选择任务类型",
            options=['binary', 'multiclass', 'regression'],
            format_func=lambda x: {'binary': '二分类', 'multiclass': '多分类', 'regression': '回归'}[x],
            index=['binary', 'multiclass', 'regression'].index(st.session_state.task_type),
        )
        st.session_state.task_type = task_type

    from src.data_exploration import TargetValidator
    is_valid, msg = TargetValidator.validate(
        st.session_state.df, st.session_state.target_col, st.session_state.task_type
    )

    if is_valid:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")

    if is_valid:
        if st.button("🚀 开始下一步：特征工程", type="primary"):
            pipeline = st.session_state.pipeline
            pipeline.load_data(st.session_state.df)
            pipeline.column_types = st.session_state.column_types
            pipeline.validate_target(st.session_state.target_col, st.session_state.task_type)
            pipeline.prepare_datasets()

            st.session_state.current_step = 1
            st.rerun()


def step_feature_engineering():
    """步骤2: 自动特征工程"""
    st.title("⚙️ 自动特征工程")
    st.markdown("系统将自动生成衍生特征，包括数值变换、分类编码、日期拆解等。")

    with st.expander("⚙️ 特征工程参数设置", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            text_strategy = st.selectbox(
                "文本特征处理策略",
                options=['tfidf', 'drop'],
                format_func=lambda x: {'tfidf': 'TF-IDF向量化', 'drop': '直接丢弃'}[x],
                index=0,
            )
            max_tfidf_features = st.slider("TF-IDF最大特征数", 10, 500, 100, 10)

        with col2:
            enable_poly_cross = st.checkbox("启用多项式交叉特征", value=True)
            n_bins = st.slider("数值分箱档数", 3, 10, 5, 1)

        with col3:
            corr_threshold = st.slider("高相关过滤阈值", 0.8, 1.0, 0.95, 0.01)

    if st.button("🔧 运行特征工程", type="primary"):
        with st.spinner("正在进行特征工程..."):
            pipeline = st.session_state.pipeline
            pipeline.column_types = st.session_state.column_types

            result = pipeline.run_feature_engineering(
                text_strategy=text_strategy,
                enable_poly_cross=enable_poly_cross,
                corr_threshold=corr_threshold,
                max_tfidf_features=max_tfidf_features,
                n_bins=n_bins,
            )
            st.session_state.feature_engineering_result = result

    if st.session_state.feature_engineering_result:
        result = st.session_state.feature_engineering_result

        st.markdown("---")
        st.subheader("📊 特征工程结果")

        col1, col2, col3 = st.columns(3)
        col1.metric("原始特征数", len(st.session_state.column_types) - 1)
        col2.metric("生成特征数", result['n_features_full'])
        col3.metric("特征增长率", f"{(result['n_features_full'] / max(len(st.session_state.column_types) - 1, 1)) * 100:.0f}%")

        st.markdown("#### 🔄 变换步骤")
        for step_name, step_info in result['transform_steps']:
            st.text(f"  • {step_name}: {step_info}")

        with st.expander("📋 查看所有特征名称"):
            st.write(result['feature_names'])

        st.markdown("---")
        st.subheader("🔍 特征交互探索")
        st.caption("选择两个特征，查看它们的交互效果及与目标变量的互信息")

        from src.feature_engineering import FeatureInteractionExplorer

        explorer = FeatureInteractionExplorer(
            st.session_state.df,
            st.session_state.column_types,
            st.session_state.target_col
        )

        available = explorer.get_available_features()
        all_features = available['numeric'] + available['categorical']

        col1, col2 = st.columns(2)
        with col1:
            feat1 = st.selectbox("选择特征1", options=all_features, key='feat1_interaction')
        with col2:
            feat2 = st.selectbox("选择特征2", options=[f for f in all_features if f != feat1], key='feat2_interaction')

        if st.button("🔬 分析交互效果", key='analyze_interaction'):
            with st.spinner("正在分析特征交互..."):
                interaction = explorer.analyze_interaction(feat1, feat2, st.session_state.task_type)

            if 'error' in interaction:
                st.error(interaction['error'])
            else:
                mi_info = interaction.get('mi_info', {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{feat1} 互信息", f"{mi_info.get('mi_col1', 0):.4f}")
                with col2:
                    st.metric(f"{feat2} 互信息", f"{mi_info.get('mi_col2', 0):.4f}")
                with col3:
                    st.metric("联合互信息(近似)", f"{mi_info.get('mi_pair', 0):.4f}")

                st.markdown("#### 📊 可视化")

                if interaction['type'] == 'numeric_numeric':
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = interaction['plot_data']['x']
                    y = interaction['plot_data']['y']
                    sample_size = min(1000, len(x))
                    indices = np.random.choice(len(x), sample_size, replace=False) if len(x) > sample_size else np.arange(len(x))
                    ax.scatter(np.array(x)[indices], np.array(y)[indices], alpha=0.6, s=20)
                    ax.set_xlabel(feat1)
                    ax.set_ylabel(feat2)
                    ax.set_title(f"{feat1} vs {feat2} 散点图 (Pearson相关系数: {interaction['correlation']:.4f})")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.info(f"**Pearson相关系数**: {interaction['correlation']:.4f}  \n**样本数**: {interaction['n_samples']}")

                elif interaction['type'] == 'numeric_categorical':
                    num_col = interaction['numeric_col']
                    cat_col = interaction['categorical_col']
                    groups = interaction['groups']

                    fig, ax = plt.subplots(figsize=(10, 6))
                    group_data = [g['values'] for g in groups.values()]
                    group_labels = list(groups.keys())
                    if len(group_labels) > 10:
                        top_groups = sorted(groups.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
                        group_data = [g['values'] for _, g in top_groups]
                        group_labels = [k for k, _ in top_groups]
                        st.caption("（仅显示样本数最多的前10个类别）")

                    bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True)
                    colors = plt.cm.Set2(np.linspace(0, 1, len(group_labels)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_xlabel(cat_col)
                    ax.set_ylabel(num_col)
                    ax.set_title(f"{num_col} 按 {cat_col} 分组箱线图 (F统计量: {interaction['f_statistic']:.4f}, p值: {interaction['p_value']:.4f})")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.info(f"**F统计量**: {interaction['f_statistic']:.4f}  \n**p值**: {interaction['p_value']:.4f}  \n**样本数**: {interaction['n_samples']}")

                    group_df = pd.DataFrame([
                        {'类别': k, '均值': v['mean'], '中位数': v['median'], '样本数': v['count']}
                        for k, v in groups.items()
                    ]).sort_values('样本数', ascending=False)
                    with st.expander("📋 查看分组统计详情"):
                        st.dataframe(group_df, use_container_width=True)

                elif interaction['type'] == 'categorical_categorical':
                    fig, ax = plt.subplots(figsize=(10, 8))
                    values = np.array(interaction['values'])
                    index = interaction['index']
                    columns = interaction['columns']

                    if len(index) > 20 or len(columns) > 20:
                        st.caption("（类别过多，仅显示部分数据）")
                        if len(index) > 20:
                            values = values[:20, :]
                            index = index[:20]
                        if len(columns) > 20:
                            values = values[:, :20]
                            columns = columns[:20]

                    sns.heatmap(
                        values,
                        annot=True,
                        fmt='d',
                        cmap='YlOrRd',
                        ax=ax,
                        xticklabels=columns,
                        yticklabels=index,
                    )
                    ax.set_xlabel(interaction['col2'])
                    ax.set_ylabel(interaction['col1'])
                    ax.set_title(f"{interaction['col1']} × {interaction['col2']} 交叉频次热力图 (卡方: {interaction['chi_square']:.4f}, p值: {interaction['p_value']:.4f})")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.info(f"**卡方统计量**: {interaction['chi_square']:.4f}  \n**p值**: {interaction['p_value']:.4f}  \n**样本数**: {interaction['n_samples']}")

        if st.button("➡️ 下一步：特征重要性评估", type="primary"):
            st.session_state.current_step = 2
            st.rerun()


def step_feature_selection():
    """步骤3: 特征重要性评估"""
    st.title("🎯 特征重要性评估")
    st.markdown("使用三种独立方法评估特征重要性，并取交集筛选最终特征集。")

    with st.expander("⚙️ 特征选择参数设置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_methods_required = st.slider(
                "至少被几种方法认为重要",
                1, 3, 2, 1,
                help="取至少被N种方法认为重要的特征"
            )
            auto_select = st.checkbox("自动选择特征数", value=True)

        with col2:
            n_estimators = st.slider("随机森林树数量", 50, 300, 100, 10)
            if not auto_select:
                n_features = st.slider("保留特征数", 5, 200, 20, 5)
            else:
                n_features = None
                auto_threshold = st.slider("累计重要性阈值", 0.5, 0.95, 0.8, 0.05)

    if st.button("🔍 运行特征重要性评估", type="primary"):
        with st.spinner("正在计算特征重要性..."):
            pipeline = st.session_state.pipeline
            result = pipeline.run_feature_selection(
                n_methods_required=n_methods_required,
                n_estimators=n_estimators,
                n_features=None if auto_select else n_features,
                auto=auto_select,
                auto_threshold=auto_threshold if auto_select else 0.8,
            )
            st.session_state.feature_selection_result = result

    if st.session_state.feature_selection_result:
        result = st.session_state.feature_selection_result

        st.markdown("---")
        st.subheader("📊 特征选择结果")

        col1, col2 = st.columns(2)
        col1.metric("原始特征数", st.session_state.feature_engineering_result['n_features_full'] if st.session_state.feature_engineering_result else 0)
        col2.metric("选中特征数", result['n_selected'])

        importances = result['importances']

        st.markdown("---")
        st.subheader("📈 三种方法重要性对比")

        top_n = min(20, result['n_selected'])

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        methods = [
            ('random_forest', '随机森林Gini重要性'),
            ('permutation', '排列重要性'),
            ('l1_regularization', 'L1正则化重要性'),
        ]

        for i, (method, title) in enumerate(methods):
            if method in importances.columns:
                data = importances[method].sort_values(ascending=False).head(top_n)
                sns.barplot(x=data.values, y=data.index, ax=axes[i], palette='viridis')
                axes[i].set_title(title, fontsize=12)
                axes[i].set_xlabel('重要性')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("🔵 选中的特征")

        with st.expander("📋 查看所有选中特征"):
            for i, feat in enumerate(result['selected_features'], 1):
                st.text(f"{i}. {feat}")

        analyzer = st.session_state.pipeline.feature_analyzer
        selector = st.session_state.pipeline.feature_selector

        if analyzer and selector:
            st.markdown("---")
            st.subheader("🎯 Venn图 - 三种方法交集")

            venn_data = selector.get_venn_data(analyzer)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("仅随机森林", len(venn_data['rf_only']))
            with col2:
                st.metric("仅排列重要性", len(venn_data['perm_only']))
            with col3:
                st.metric("仅L1正则化", len(venn_data['l1_only']))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("随机森林+排列", len(venn_data['rf_perm']))
            with col2:
                st.metric("随机森林+L1", len(venn_data['rf_l1']))
            with col3:
                st.metric("排列+L1", len(venn_data['perm_l1']))

            st.metric("✅ 三种方法都选中", len(venn_data['all_three']))

        if st.button("➡️ 下一步：模型选择", type="primary"):
            st.session_state.current_step = 3
            st.rerun()


def step_model_selection():
    """步骤4: 自动模型选择"""
    st.title("🤖 自动模型选择")
    st.markdown("使用贝叶斯优化对多种模型进行超参搜索，找出最佳模型配置。")

    with st.expander("⚙️ 模型选择参数设置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.slider("每个模型试验次数", 10, 100, 30, 5)
        with col2:
            cv_folds = st.slider("交叉验证折数", 3, 10, 5, 1)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    col1, col2 = st.columns([1, 1])
    with col1:
        start_button = st.button("🚀 开始模型搜索", type="primary", disabled=st.session_state.is_running)
    with col2:
        stop_button = st.button("⏹️ 停止搜索", disabled=not st.session_state.is_running)

    if start_button:
        st.session_state.is_running = True

        pipeline = st.session_state.pipeline

        progress_bar = st.progress(0)

        def progress_callback(msg):
            status_placeholder.text(msg)

        with st.spinner("正在进行模型搜索，请耐心等待..."):
            try:
                result = pipeline.run_model_selection(
                    n_trials=n_trials,
                    cv=cv_folds,
                    progress_callback=progress_callback,
                )
                st.session_state.model_selection_result = result
            except Exception as e:
                st.error(f"模型搜索出错: {str(e)}")
            finally:
                st.session_state.is_running = False

        st.rerun()

    if stop_button:
        st.session_state.pipeline.stop_model_selection()
        st.session_state.is_running = False
        st.info("已停止搜索，保留当前最优结果")

    if st.session_state.model_selection_result:
        result = st.session_state.model_selection_result
        results_df = result['results_df']

        st.markdown("---")
        st.subheader("🏆 模型搜索结果")

        st.success(f"🥇 最佳模型: **{result['best_model_name']}**，得分: **{result['best_score']:.4f}**")

        st.markdown("#### 📊 所有模型对比")

        if not results_df.empty:
            display_df = results_df.copy()

            if 'cv_mean' in display_df.columns:
                display_df['cv_mean'] = display_df['cv_mean'].round(4)
            if 'cv_std' in display_df.columns:
                display_df['cv_std'] = display_df['cv_std'].round(4)
            if 'train_time' in display_df.columns:
                display_df['train_time'] = display_df['train_time'].round(2)
            if 'best_params' in display_df.columns:
                display_df['best_params'] = display_df['best_params'].apply(
                    lambda x: ', '.join([f"{k}={v}" for k, v in x.items()][:5]) if isinstance(x, dict) else str(x)
                )

            st.dataframe(display_df, use_container_width=True)

            st.markdown("#### 📈 模型性能对比图")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=results_df,
                x='model_name',
                y='cv_mean',
                yerr=results_df['cv_std'],
                ax=ax,
                palette='Set2',
                capsize=0.1,
            )
            ax.set_xlabel('模型')
            ax.set_ylabel('CV得分')
            ax.set_title('各模型交叉验证得分对比')
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if st.button("➡️ 下一步：模型诊断", type="primary"):
            st.session_state.current_step = 4
            st.rerun()


def step_model_diagnosis():
    """步骤5: 模型对比与诊断"""
    st.title("📊 模型对比与诊断")
    st.markdown("对最优模型进行详细诊断，评估模型性能和可靠性。")

    if st.button("🔍 运行模型诊断", type="primary"):
        with st.spinner("正在进行模型诊断..."):
            pipeline = st.session_state.pipeline
            result = pipeline.run_diagnosis()
            st.session_state.diagnosis_result = result

    if st.session_state.diagnosis_result:
        result = st.session_state.diagnosis_result
        metrics = result['metrics']

        st.markdown("---")
        st.subheader("📈 模型性能指标")

        if result['is_overfitting']:
            st.warning("⚠️ **可能存在过拟合**")
            st.info("💡 建议:")
            for suggestion in result['overfitting_suggestions']:
                st.text(f"  • {suggestion}")

        task_type = st.session_state.task_type

        if task_type in ['binary', 'multiclass']:
            col1, col2, col3 = st.columns(3)
            col1.metric("训练集得分", f"{metrics.get('train_score', 0):.4f}")
            col2.metric("测试集得分", f"{metrics.get('test_score', 0):.4f}")
            col3.metric("准确率", f"{metrics.get('accuracy', 0):.4f}")

            if 'roc_auc' in metrics:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

            st.markdown("---")
            st.subheader("📊 混淆矩阵")

            if 'confusion_matrix' in metrics and metrics['confusion_matrix']:
                cm = np.array(metrics['confusion_matrix'])
                classes = metrics.get('classes', [])

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    xticklabels=classes if classes else 'auto',
                    yticklabels=classes if classes else 'auto',
                )
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if 'classification_report' in metrics and metrics['classification_report']:
                st.markdown("---")
                st.subheader("📋 分类报告")
                report = metrics['classification_report']
                if isinstance(report, dict):
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)

            if 'roc_curve' in metrics and task_type == 'binary':
                st.markdown("---")
                st.subheader("📉 ROC曲线")

                fig, ax = plt.subplots(figsize=(8, 6))
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                roc_auc = metrics.get('roc_auc', 0)

                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlabel('假阳性率 (FPR)')
                ax.set_ylabel('真阳性率 (TPR)')
                ax.set_title('ROC曲线')
                ax.legend(loc='lower right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if 'pr_curve' in metrics and task_type == 'binary':
                st.markdown("---")
                st.subheader("📉 Precision-Recall曲线")

                fig, ax = plt.subplots(figsize=(8, 6))
                precision = metrics['pr_curve']['precision']
                recall = metrics['pr_curve']['recall']
                ap = metrics['pr_curve']['average_precision']

                ax.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {ap:.4f})')
                ax.set_xlabel('召回率 (Recall)')
                ax.set_ylabel('精确率 (Precision)')
                ax.set_title('Precision-Recall曲线')
                ax.legend(loc='lower left')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            col2.metric("R²", f"{metrics.get('r2', 0):.4f}")
            col3.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            col4.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

            st.markdown("---")
            st.subheader("📉 残差图")

            if 'y_true' in metrics and 'y_pred' in metrics:
                y_true = np.array(metrics['y_true'])
                y_pred = np.array(metrics['y_pred'])
                residuals = y_true - y_pred

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
                axes[0].axhline(y=0, color='r', linestyle='--')
                axes[0].set_xlabel('预测值')
                axes[0].set_ylabel('残差')
                axes[0].set_title('残差 vs 预测值')

                axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                axes[1].set_xlabel('残差')
                axes[1].set_ylabel('频数')
                axes[1].set_title('残差分布直方图')

                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=axes[2])
                axes[2].set_title('QQ图')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        shap_data = result.get('shap_data')
        if shap_data:
            st.markdown("---")
            st.subheader("🔵 SHAP特征贡献（前10个特征）")

            try:
                import shap
                top_features = shap_data.get('top_features', [])
                top_importances = shap_data.get('top_importances', [])

                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_features))
                ax.barh(y_pos, top_importances, color='skyblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features)
                ax.invert_yaxis()
                ax.set_xlabel('平均|SHAP值|')
                ax.set_title('Top 10 特征重要性 (SHAP)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"SHAP可视化暂不可用: {str(e)}")

        learning_curve = result.get('learning_curve')
        if learning_curve:
            st.markdown("---")
            st.subheader("📈 学习曲线分析")
            st.caption("使用不同比例的训练数据进行交叉验证，判断模型欠拟合/过拟合状态")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最终训练得分", f"{learning_curve['final_train_score']:.4f}")
            with col2:
                st.metric("最终验证得分", f"{learning_curve['final_test_score']:.4f}")
            with col3:
                st.metric("训练-验证差距", f"{learning_curve['score_gap']:.4f}")

            status_msgs = []
            if learning_curve.get('is_underfitting'):
                status_msgs.append("⚠️ 可能欠拟合")
            if learning_curve.get('is_overfitting'):
                status_msgs.append("⚠️ 可能过拟合")
            if learning_curve.get('needs_more_data'):
                status_msgs.append("📈 增加数据可能有帮助")
            if not status_msgs:
                status_msgs.append("✅ 模型状态良好")

            st.info(" | ".join(status_msgs))

            fig, ax = plt.subplots(figsize=(10, 6))

            train_sizes_pct = [int(s * 100) for s in learning_curve['train_sizes']]
            train_mean = learning_curve['train_scores_mean']
            train_std = learning_curve['train_scores_std']
            test_mean = learning_curve['test_scores_mean']
            test_std = learning_curve['test_scores_std']

            ax.plot(train_sizes_pct, train_mean, 'o-', color='blue', label='训练集得分', linewidth=2)
            ax.fill_between(train_sizes_pct,
                         np.array(train_mean) - np.array(train_std),
                         np.array(train_mean) + np.array(train_std),
                         alpha=0.15, color='blue')

            ax.plot(train_sizes_pct, test_mean, 's-', color='orange', label='验证集得分', linewidth=2)
            ax.fill_between(train_sizes_pct,
                         np.array(test_mean) - np.array(test_std),
                         np.array(test_mean) + np.array(test_std),
                         alpha=0.15, color='orange')

            ax.set_xlabel('训练数据比例 (%)')
            ax.set_ylabel(f"得分 ({learning_curve.get('scoring', 'score')})")
            ax.set_title('学习曲线 - 训练集 vs 验证集')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            all_scores = train_mean + test_mean
            min_score = min(all_scores)
            ax.set_ylim([max(0, min_score - 0.1), 1.05])

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### 💡 分析建议")
            for suggestion in learning_curve.get('suggestions', []):
                st.text(f"  • {suggestion}")

            with st.expander("📋 查看详细数据"):
                lc_df = pd.DataFrame({
                    '训练比例(%)': train_sizes_pct,
                    '训练样本数': learning_curve['train_sample_sizes'],
                    '训练得分': learning_curve['train_scores_mean'],
                    '训练标准差': learning_curve['train_scores_std'],
                    '验证得分': learning_curve['test_scores_mean'],
                    '验证标准差': learning_curve['test_scores_std'],
                })
                st.dataframe(lc_df, use_container_width=True)

        if st.button("➡️ 下一步：模型可解释性分析", type="primary"):
            st.session_state.current_step = 5
            st.rerun()


def step_interpretability():
    """步骤6: 模型可解释性分析"""
    st.title("🔬 模型可解释性分析")
    st.markdown("从多个角度理解模型决策逻辑，包括局部解释、全局解释、对抗性检测和报告导出。")

    pipeline = st.session_state.pipeline
    X = pipeline.X_selected if pipeline.is_feature_selected else pipeline.X_full
    y = pipeline.y_full
    best_model = None
    best_model_name = 'Unknown'
    if pipeline.model_selector:
        best_model = pipeline.model_selector.get_best_model()
        best_model_name = pipeline.model_selector.get_best_model_name()

    if X is None or best_model is None:
        st.warning("⚠️ 请先完成模型选择步骤")
        return

    feature_names = list(X.columns)

    if not st.session_state.interpretability_result:
        col_run1, col_run2 = st.columns([3, 1])
        with col_run1:
            output_dir = st.text_input("分析结果输出目录", value="./output", key="interp_outdir")
        with col_run2:
            st.write("")
            st.write("")
            run_full_btn = st.button("🚀 运行完整可解释性分析", type="primary")

        if run_full_btn:
            with st.spinner("正在进行可解释性分析（可能需要几分钟）..."):
                result = pipeline.run_interpretability(output_dir=output_dir)
                st.session_state.interpretability_result = result
                st.success("✅ 完整分析完成！")
                st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 局部解释面板",
        "🌐 全局解释面板",
        "🛡️ 对抗性解释检测",
        "📄 解释报告导出",
    ])

    with tab1:
        st.header("🔍 局部解释面板")
        local_subtab1, local_subtab2 = st.tabs(["📊 单样本解释", "🔄 样本对比模式"])

        with local_subtab1:
            st.caption("选择一条样本，同时展示三种局部解释结果对比")

            col1, col2 = st.columns([1, 3])
            with col1:
                sample_idx = st.number_input(
                    "选择样本索引",
                    min_value=0,
                    max_value=max(0, len(X) - 1),
                    value=0,
                    step=1,
                    key="local_sample_idx",
                )

            with col2:
                st.markdown("**样本特征值预览:**")
                sample_row = X.iloc[[sample_idx]]
                st.dataframe(sample_row.T, use_container_width=True)

            if st.button("🔬 分析该样本", key="run_local"):
                with st.spinner("正在计算局部解释..."):
                    local = LocalInterpreter(
                        best_model, X, pipeline.task_type, feature_names, pipeline.random_state
                    )
                    shap_res = local.explain_shap(sample_idx)
                    lime_res = local.explain_lime(sample_idx)
                    ice_res = local.explain_ice(sample_idx)

                    if 'error' in shap_res:
                        st.error(f"SHAP解释出错: {shap_res['error']}")
                    if 'error' in lime_res:
                        st.error(f"LIME解释出错: {lime_res['error']}")
                    if 'error' in ice_res:
                        st.error(f"ICE解释出错: {ice_res['error']}")

                    if 'error' not in shap_res and 'error' not in lime_res and 'error' not in ice_res:
                        consistency = local.compute_consistency_score(shap_res, lime_res, ice_res)

                        st.markdown("---")
                        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                        col_c1.metric("SHAP vs LIME (τ)", f"{consistency['kendall_shap_lime']:.3f}")
                        col_c2.metric("SHAP vs ICE (τ)", f"{consistency['kendall_shap_ice']:.3f}")
                        col_c3.metric("LIME vs ICE (τ)", f"{consistency['kendall_lime_ice']:.3f}")
                        col_c4.metric(
                            "一致性评分",
                            f"{consistency['mean_consistency']:.3f}",
                            delta=consistency['label'],
                            delta_color='inverse' if consistency['is_conflict'] else 'normal',
                        )
                        if consistency['is_conflict']:
                            st.warning("⚠️ **解释冲突**：三种方法的特征重要性排序一致性低于0.6，建议谨慎分析该样本的模型决策。")

                        st.markdown("---")
                        col_s1, col_s2, col_s3 = st.columns(3)

                        with col_s1:
                            st.subheader("SHAP瀑布图")
                            if 'feature_contributions' in shap_res:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                contribs = shap_res['feature_contributions'][:10]
                                feats = [f"{c['feature']}={c['value']:.2f}" for c in contribs]
                                vals = [c['shap_value'] for c in contribs]
                                base_val = shap_res.get('base_value', 0)

                                cum_vals = [base_val]
                                for v in vals:
                                    cum_vals.append(cum_vals[-1] + v)

                                colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals]
                                y_pos = np.arange(len(contribs))

                                for i, (feat, val, cum, color) in enumerate(zip(feats, vals, cum_vals[:-1], colors)):
                                    ax.barh(i, val, left=cum, color=color, alpha=0.8, height=0.6)

                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(feats)
                                ax.axvline(x=base_val, color='gray', linestyle='--', alpha=0.5, label=f'Base: {base_val:.3f}')
                                ax.axvline(x=cum_vals[-1], color='blue', linestyle='-', alpha=0.7, label=f'Pred: {cum_vals[-1]:.3f}')
                                ax.set_xlabel('SHAP值')
                                ax.set_title(f'SHAP瀑布图 - 样本 #{sample_idx}')
                                ax.legend()
                                ax.invert_yaxis()
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                        with col_s2:
                            st.subheader("LIME特征权重")
                            if 'feature_weights' in lime_res:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                weights = lime_res['feature_weights'][:10]
                                feats = [w['feature'] for w in weights]
                                vals = [w['weight'] for w in weights]
                                colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals]

                                y_pos = np.arange(len(feats))
                                ax.barh(y_pos, vals, color=colors, alpha=0.8, height=0.6)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(feats)
                            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                            ax.set_xlabel('特征权重')
                            ax.set_title(f'LIME特征权重 - 样本 #{sample_idx}')
                            ax.invert_yaxis()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    with col_s3:
                        st.subheader("ICE边际效应曲线")
                        if 'ice_curves' in ice_res:
                            ice_data = ice_res['ice_curves']
                            if ice_data:
                                n_feats = min(3, len(ice_data))
                                fig, axes = plt.subplots(n_feats, 1, figsize=(8, 3 * n_feats))
                                if n_feats == 1:
                                    axes = [axes]
                                for i, (feat, ice_d) in enumerate(list(ice_data.items())[:n_feats]):
                                    axes[i].plot(ice_d['grid_values'], ice_d['predictions'], 'b-', linewidth=2)
                                    axes[i].axvline(
                                        x=ice_d['original_value'],
                                        color='red',
                                        linestyle='--',
                                        label=f'原值: {ice_d["original_value"]:.2f}',
                                    )
                                    axes[i].set_xlabel(feat)
                                    axes[i].set_ylabel('预测值')
                                    axes[i].set_title(f'{feat} - 边际效应 (Δ={ice_d["marginal_effect"]:.4f})')
                                    axes[i].legend()
                                    axes[i].grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                    st.markdown("---")
                    st.subheader("📊 TOP-5 特征重要性排序对比")
                    compare_df = pd.DataFrame({
                        'SHAP': shap_res.get('top_features', []),
                        'LIME': lime_res.get('top_features', []),
                        'ICE': ice_res.get('top_features', []),
                    }, index=[f'Rank {i+1}' for i in range(5)])
                    st.dataframe(compare_df, use_container_width=True)

        with local_subtab2:
            st.caption("选择两条样本（建议一条正例一条反例），并排展示解释结果并分析决策差异")

            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                sample_idx1 = st.number_input(
                    "样本1索引（建议正例）",
                    min_value=0,
                    max_value=max(0, len(X) - 1),
                    value=0,
                    step=1,
                    key="compare_sample1",
                )
                st.markdown("**样本1特征值预览:**")
                sample_row1 = X.iloc[[sample_idx1]]
                st.dataframe(sample_row1.T, use_container_width=True)

            with col_comp2:
                sample_idx2 = st.number_input(
                    "样本2索引（建议反例）",
                    min_value=0,
                    max_value=max(0, len(X) - 1),
                    value=min(1, len(X) - 1),
                    step=1,
                    key="compare_sample2",
                )
                st.markdown("**样本2特征值预览:**")
                sample_row2 = X.iloc[[sample_idx2]]
                st.dataframe(sample_row2.T, use_container_width=True)

            if st.button("🔬 对比分析两个样本", key="run_compare"):
                if sample_idx1 == sample_idx2:
                    st.warning("⚠️ 两个样本索引相同，请选择不同的样本进行对比。")
                else:
                    with st.spinner("正在计算样本对比解释..."):
                        local = LocalInterpreter(
                            best_model, X, pipeline.task_type, feature_names, pipeline.random_state
                        )
                        compare_res = local.explain_compare(sample_idx1, sample_idx2)

                        if 'error' in compare_res:
                            st.error(f"对比分析出错: {compare_res['error']}")
                        else:
                            s1 = compare_res['sample1']
                            s2 = compare_res['sample2']
                            diff = compare_res.get('decision_difference')

                            st.markdown("---")
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                            pred1 = s1.get('prediction', 0)
                            pred2 = s2.get('prediction', 0)
                            pred_diff = abs(pred1 - pred2)
                            n_opposite = diff.get('n_opposite_features', 0) if diff else 0

                            col_m1.metric(f"样本 #{s1['idx']} 预测值", f"{pred1:.4f}")
                            col_m2.metric(f"样本 #{s2['idx']} 预测值", f"{pred2:.4f}")
                            col_m3.metric("预测值差异", f"{pred_diff:.4f}")
                            col_m4.metric("贡献方向相反特征数", n_opposite)

                            if diff:
                                st.info(diff.get('summary', ''))
                            else:
                                st.info("💡 SHAP解释器不可用，无法进行详细的决策差异分析，但仍可查看基础对比结果。")

                            st.markdown("---")
                            st.subheader("📊 SHAP瀑布图对比")
                            col_w1, col_w2 = st.columns(2)

                            with col_w1:
                                shap1 = s1.get('shap', {})
                                if 'feature_contributions' in shap1:
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    contribs1 = shap1['feature_contributions'][:10]
                                    feats1 = [f"{c['feature']}={c['value']:.2f}" for c in contribs1]
                                    vals1 = [c['shap_value'] for c in contribs1]
                                    base_val1 = shap1.get('base_value', 0)

                                    cum_vals1 = [base_val1]
                                    for v in vals1:
                                        cum_vals1.append(cum_vals1[-1] + v)

                                    colors1 = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals1]
                                    y_pos1 = np.arange(len(contribs1))

                                    for i, (feat, val, cum, color) in enumerate(zip(feats1, vals1, cum_vals1[:-1], colors1)):
                                        ax.barh(i, val, left=cum, color=color, alpha=0.8, height=0.6)

                                    ax.set_yticks(y_pos1)
                                    ax.set_yticklabels(feats1)
                                    ax.axvline(x=base_val1, color='gray', linestyle='--', alpha=0.5, label=f'Base: {base_val1:.3f}')
                                    ax.axvline(x=cum_vals1[-1], color='blue', linestyle='-', alpha=0.7, label=f'Pred: {cum_vals1[-1]:.3f}')
                                    ax.set_xlabel('SHAP值')
                                    ax.set_title(f'样本 #{s1["idx"]} SHAP瀑布图')
                                    ax.legend()
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.info("💡 样本1 SHAP解释不可用")

                            with col_w2:
                                shap2 = s2.get('shap', {})
                                if 'feature_contributions' in shap2:
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    contribs2 = shap2['feature_contributions'][:10]
                                    feats2 = [f"{c['feature']}={c['value']:.2f}" for c in contribs2]
                                    vals2 = [c['shap_value'] for c in contribs2]
                                    base_val2 = shap2.get('base_value', 0)

                                    cum_vals2 = [base_val2]
                                    for v in vals2:
                                        cum_vals2.append(cum_vals2[-1] + v)

                                    colors2 = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals2]
                                    y_pos2 = np.arange(len(contribs2))

                                    for i, (feat, val, cum, color) in enumerate(zip(feats2, vals2, cum_vals2[:-1], colors2)):
                                        ax.barh(i, val, left=cum, color=color, alpha=0.8, height=0.6)

                                    ax.set_yticks(y_pos2)
                                    ax.set_yticklabels(feats2)
                                    ax.axvline(x=base_val2, color='gray', linestyle='--', alpha=0.5, label=f'Base: {base_val2:.3f}')
                                    ax.axvline(x=cum_vals2[-1], color='blue', linestyle='-', alpha=0.7, label=f'Pred: {cum_vals2[-1]:.3f}')
                                    ax.set_xlabel('SHAP值')
                                    ax.set_title(f'样本 #{s2["idx"]} SHAP瀑布图')
                                    ax.legend()
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.info("💡 样本2 SHAP解释不可用")

                        st.markdown("---")
                        st.subheader("📊 LIME特征权重对比")
                        col_l1, col_l2 = st.columns(2)

                        with col_l1:
                            lime1 = s1.get('lime', {})
                            if 'feature_weights' in lime1:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                weights1 = lime1['feature_weights'][:10]
                                feats1 = [w['feature'] for w in weights1]
                                vals1 = [w['weight'] for w in weights1]
                                colors1 = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals1]

                                y_pos1 = np.arange(len(feats1))
                                ax.barh(y_pos1, vals1, color=colors1, alpha=0.8, height=0.6)
                                ax.set_yticks(y_pos1)
                                ax.set_yticklabels(feats1)
                                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                                ax.set_xlabel('特征权重')
                                ax.set_title(f'样本 #{s1["idx"]} LIME特征权重')
                                ax.invert_yaxis()
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("💡 样本1 LIME解释不可用")

                        with col_l2:
                            lime2 = s2.get('lime', {})
                            if 'feature_weights' in lime2:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                weights2 = lime2['feature_weights'][:10]
                                feats2 = [w['feature'] for w in weights2]
                                vals2 = [w['weight'] for w in weights2]
                                colors2 = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals2]

                                y_pos2 = np.arange(len(feats2))
                                ax.barh(y_pos2, vals2, color=colors2, alpha=0.8, height=0.6)
                                ax.set_yticks(y_pos2)
                                ax.set_yticklabels(feats2)
                                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                                ax.set_xlabel('特征权重')
                                ax.set_title(f'样本 #{s2["idx"]} LIME特征权重')
                                ax.invert_yaxis()
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("💡 样本2 LIME解释不可用")

                        if diff and diff.get('opposite_features'):
                            st.markdown("---")
                            st.subheader("⚡ 决策差异分析")
                            st.caption("以下特征在两个样本间的SHAP贡献方向相反（一个正贡献一个负贡献），按差异绝对值从大到小排序：")

                            diff_df = pd.DataFrame(diff['opposite_features'])[
                                ['feature', 'sample1_shap', 'sample2_shap', 'diff_abs', 'direction']
                            ]
                            diff_df.columns = ['特征', '样本1 SHAP值', '样本2 SHAP值', '差异绝对值', '方向变化']
                            st.dataframe(diff_df, use_container_width=True)

                            st.info(f"💡 **分析提示**: 这 {diff['n_opposite_features']} 个特征是导致两个样本预测结果差异的关键因素。"
                                    f" 它们在样本1和样本2中起到了相反的推动作用。")

    with tab2:
        st.header("🌐 全局解释面板")
        st.caption("对整个验证集进行聚合分析")

        global_interp = GlobalInterpreter(
            best_model, X, y, pipeline.task_type, feature_names, pipeline.random_state
        )

        if st.button("📊 计算全局解释", key="run_global"):
            with st.spinner("正在计算全局解释指标..."):
                shap_global = global_interp.shap_global_importance()
                perm_global = global_interp.permutation_importance()
                stability = global_interp.feature_stability_analysis(shap_global)

                if 'error' not in shap_global:
                    st.session_state.interpretability_global_shap = shap_global
                if 'error' not in perm_global:
                    st.session_state.interpretability_global_perm = perm_global
                if 'error' not in stability:
                    st.session_state.interpretability_stability = stability

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("SHAP全局重要性")
            shap_global = st.session_state.get('interpretability_global_shap')
            if shap_global and 'importances' in shap_global:
                top_n = min(15, len(shap_global['importances']))
                top_data = shap_global['importances'][:top_n]
                fig, ax = plt.subplots(figsize=(10, 8))
                feats = [x['feature'] for x in reversed(top_data)]
                vals = [x['mean_abs_shap'] for x in reversed(top_data)]
                colors = ['#e74c3c' if x.get('is_unstable', False) else '#3498db' for x in reversed(top_data)]
                ax.barh(feats, vals, color=colors, alpha=0.8)
                ax.set_xlabel('平均 |SHAP值|')
                ax.set_title(f'Top {top_n} 特征 SHAP 全局重要性（红色=不稳定）')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("请先点击『计算全局解释』按钮")

        with col_g2:
            st.subheader("排列重要性")
            perm_global = st.session_state.get('interpretability_global_perm')
            if perm_global and 'importances' in perm_global:
                top_n = min(15, len(perm_global['importances']))
                top_data = perm_global['importances'][:top_n]
                fig, ax = plt.subplots(figsize=(10, 8))
                feats = [x['feature'] for x in reversed(top_data)]
                vals = [x['importance_mean'] for x in reversed(top_data)]
                errs = [x['importance_std'] for x in reversed(top_data)]
                ax.barh(feats, vals, xerr=errs, color='#9b59b6', alpha=0.8, capsize=3)
                ax.set_xlabel(f"重要性 ({perm_global.get('scoring', '')})")
                ax.set_title(f'Top {top_n} 特征排列重要性')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("请先点击『计算全局解释』按钮")

        st.markdown("---")
        st.subheader("📊 特征稳定性分析")
        stability = st.session_state.get('interpretability_stability')
        if stability and 'stability_summary' in stability:
            stab = stability['stability_summary']
            col_st1, col_st2, col_st3, col_st4 = st.columns(4)
            col_st1.metric("SHAP标准差均值", f"{stab['mean_shap_std']:.4f}")
            col_st2.metric("SHAP标准差中位数", f"{stab['median_shap_std']:.4f}")
            col_st3.metric("不稳定特征数", stab['n_unstable'])
            col_st4.metric("稳定特征数", stab['n_stable'])

            col_un, col_st = st.columns(2)
            with col_un:
                st.markdown("#### ⚠️ 不稳定特征（对不同样本贡献差异大）")
                if stability.get('unstable_features'):
                    un_df = pd.DataFrame(stability['unstable_features'])[['feature', 'mean_abs_shap', 'shap_std']]
                    un_df.columns = ['特征', 'mean(|SHAP|)', 'SHAP标准差']
                    st.dataframe(un_df, use_container_width=True)
                else:
                    st.success("✅ 无明显不稳定特征")
            with col_st:
                st.markdown("#### ✅ 稳定特征")
                if stability.get('stable_features'):
                    st_df = pd.DataFrame(stability['stable_features'][:10])[['feature', 'mean_abs_shap', 'shap_std']]
                    st_df.columns = ['特征', 'mean(|SHAP|)', 'SHAP标准差']
                    st.dataframe(st_df, use_container_width=True)
        else:
            st.info("请先点击『计算全局解释』按钮")

        st.markdown("---")
        st.subheader("📈 PDP偏依赖图")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pdp_feat1 = st.selectbox("选择特征1（用于一维PDP或二维PDP）", options=numeric_features, key="pdp_f1")
            pdp_mode = st.radio("PDP模式", options=["一维PDP", "二维PDP"], horizontal=True)
        with col_p2:
            pdp_feat2 = None
            if pdp_mode == "二维PDP":
                pdp_feat2 = st.selectbox(
                    "选择特征2（用于二维PDP）",
                    options=[f for f in numeric_features if f != pdp_feat1],
                    key="pdp_f2",
                )

        if st.button("📈 绘制PDP", key="run_pdp"):
            with st.spinner("正在计算偏依赖图..."):
                if pdp_mode == "一维PDP":
                    top_models = []
                    if pipeline.model_selector:
                        top_models_raw = pipeline.model_selector.get_top_models(n=3)
                        top_models = [(name, model) for name, model, _ in top_models_raw]

                    if len(top_models) > 1:
                        pdp_res = global_interp.compute_pdp_multi(pdp_feat1, top_models)
                        if 'error' not in pdp_res and 'model_results' in pdp_res:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
                            markers = ['o', 's', '^', 'D', 'v']

                            for i, model_res in enumerate(pdp_res['model_results']):
                                color = colors[i % len(colors)]
                                marker = markers[i % len(markers)]
                                ax.plot(
                                    model_res['grid_values'],
                                    model_res['partial_dependence'],
                                    color=color,
                                    linewidth=2.5,
                                    marker=marker,
                                    label=model_res['model_name'],
                                    alpha=0.8
                                )

                            ax.set_xlabel(pdp_feat1)
                            ax.set_ylabel('预测值（偏依赖）')
                            ax.set_title(f'一维偏依赖图对比（TOP-{pdp_res["n_models"]}模型）- {pdp_feat1}')
                            ax.grid(True, alpha=0.3)
                            ax.legend(title='模型')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.info(f"💡 图中展示了{pdp_res['n_models']}个模型对同一特征的偏依赖模式。"
                                    f" 不同颜色的曲线代表不同模型，可以直观对比它们的决策逻辑差异。")
                        else:
                            st.error(f"多模型PDP计算失败: {pdp_res.get('error', '未知错误')}")
                    else:
                        pdp_res = global_interp.compute_pdp(pdp_feat1)
                        if 'error' not in pdp_res:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(pdp_res['grid_values'], pdp_res['partial_dependence'], 'r-', linewidth=2.5, marker='o')
                            ax.fill_between(pdp_res['grid_values'], pdp_res['partial_dependence'], alpha=0.15, color='red')
                            ax.set_xlabel(pdp_feat1)
                            ax.set_ylabel('预测值（偏依赖）')
                            ax.set_title(f'一维偏依赖图 - {pdp_feat1}')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.error(f"PDP计算失败: {pdp_res.get('error', '未知错误')}")
                else:
                    pdp_res = global_interp.compute_pdp_2d(pdp_feat1, pdp_feat2)
                    if 'error' not in pdp_res:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        X_grid, Y_grid = np.meshgrid(pdp_res['grid_x'], pdp_res['grid_y'])
                        Z = np.array(pdp_res['z_values']).T
                        cs = ax.contourf(X_grid, Y_grid, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
                        fig.colorbar(cs, ax=ax, label='预测值')
                        cs2 = ax.contour(X_grid, Y_grid, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)
                        ax.clabel(cs2, inline=True, fontsize=8)
                        ax.set_xlabel(pdp_feat1)
                        ax.set_ylabel(pdp_feat2)
                        ax.set_title(f'二维偏依赖等高线图（最优模型）- {pdp_feat1} vs {pdp_feat2}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.error(f"2D PDP计算失败: {pdp_res.get('error', '未知错误')}")

    with tab3:
        st.header("🛡️ 对抗性解释检测")
        st.caption("对选定样本做微小扰动，检查解释是否稳定")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            adv_sample_idx = st.number_input(
                "选择样本索引进行单样本检测",
                min_value=0,
                max_value=max(0, len(X) - 1),
                value=0,
                step=1,
                key="adv_sample_idx",
            )
        with col_a2:
            n_batch = st.slider("批量检测样本数", min_value=3, max_value=min(20, len(X)), value=5, step=1)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔬 单样本检测", key="run_adv_single"):
                with st.spinner("正在进行对抗性检测..."):
                    adv = AdversarialExplainer(
                        best_model, X, pipeline.task_type, feature_names,
                        pipeline.column_types, pipeline.random_state,
                    )
                    result = adv.detect_sensitivity(adv_sample_idx)
                    if 'error' in result:
                        st.error(f"检测失败: {result['error']}")
                    else:
                        st.session_state.adv_single_result = result
                        st.rerun()

        with col_b2:
            if st.button("📊 批量检测", key="run_adv_batch"):
                with st.spinner("正在进行批量对抗性检测..."):
                    adv = AdversarialExplainer(
                        best_model, X, pipeline.task_type, feature_names,
                        pipeline.column_types, pipeline.random_state,
                    )
                    result = adv.batch_detect(n_samples=n_batch)
                    st.session_state.adv_batch_result = result
                    st.rerun()

        adv_single = st.session_state.get('adv_single_result')
        if adv_single and 'error' not in adv_single:
            st.markdown("---")
            st.subheader(f"单样本检测结果 - 样本 #{adv_single['sample_idx']}")

            label_color = 'red' if adv_single['is_sensitive'] else 'green'
            st.markdown(f"### 状态: :{label_color}[{adv_single['label']}]")

            col_a_s1, col_a_s2, col_a_s3, col_a_s4 = st.columns(4)
            col_a_s1.metric("Mean Kendall τ", f"{adv_single['mean_kendall_tau']:.3f}")
            col_a_s2.metric("SHAP τ", f"{adv_single['shap_tau']:.3f}")
            col_a_s3.metric("LIME τ", f"{adv_single['lime_tau']:.3f}")
            col_a_s4.metric("ICE τ", f"{adv_single['ice_tau']:.3f}")

            col_a_p1, col_a_p2, col_a_p3 = st.columns(3)
            col_a_p1.metric("原始预测", f"{adv_single['original_prediction']:.4f}")
            col_a_p2.metric("扰动后预测", f"{adv_single['perturbed_prediction']:.4f}")
            col_a_p3.metric("预测变化量", f"{adv_single['prediction_change']:.4f}")

            trigger_feat = adv_single.get('trigger_feature')
            if trigger_feat:
                st.markdown(f"#### 🎯 归因不稳定特征: **{trigger_feat}**")
                st.info(f"💡 特征 **{trigger_feat}** 的扰动导致了解释排序变化最大（Kendall τ 下降 {adv_single.get('max_tau_drop', 0):.3f}），"
                        f"是导致该样本解释敏感的主要原因。")

            if adv_single['is_sensitive']:
                st.warning("⚠️ **解释敏感样本**：微小扰动导致解释大幅变动，该样本的模型预测可能不可靠。")

            st.markdown("#### TOP-3 特征排序对比")
            rank_df = pd.DataFrame({
                '方法': ['SHAP', 'LIME', 'ICE'],
                '原始TOP-3': [
                    str(adv_single['original_top3']['shap']),
                    str(adv_single['original_top3']['lime']),
                    str(adv_single['original_top3']['ice']),
                ],
                '扰动后TOP-3': [
                    str(adv_single['perturbed_top3']['shap']),
                    str(adv_single['perturbed_top3']['lime']),
                    str(adv_single['perturbed_top3']['ice']),
                ],
            })
            st.dataframe(rank_df, use_container_width=True)

        adv_batch = st.session_state.get('adv_batch_result')
        if adv_batch:
            st.markdown("---")
            st.subheader("批量检测结果汇总")
            col_ab1, col_ab2, col_ab3, col_ab4 = st.columns(4)
            col_ab1.metric("检测样本数", adv_batch['total_samples'])
            col_ab2.metric("敏感样本数", adv_batch['n_sensitive'])
            col_ab3.metric("稳定样本数", adv_batch['n_stable'])
            col_ab4.metric("通过率", f"{adv_batch['adversarial_pass_rate']*100:.1f}%")

            if adv_batch.get('sample_results'):
                with st.expander("📋 查看详细检测结果"):
                    detail_rows = []
                    for r in adv_batch['sample_results']:
                        trigger_feat = r.get('trigger_feature', '-')
                        if trigger_feat is None:
                            trigger_feat = '-'
                        detail_rows.append({
                            '样本索引': f"#{r['sample_idx']}",
                            '状态': r['label'],
                            '触发特征': trigger_feat,
                            'Mean τ': f"{r['mean_kendall_tau']:.3f}",
                            'SHAP τ': f"{r['shap_tau']:.3f}",
                            'LIME τ': f"{r['lime_tau']:.3f}",
                            'ICE τ': f"{r['ice_tau']:.3f}",
                            '预测变化': f"{r['prediction_change']:.4f}",
                        })
                    detail_df = pd.DataFrame(detail_rows)
                    st.dataframe(detail_df, use_container_width=True)

                    if any(r.get('trigger_feature') for r in adv_batch['sample_results'] if r.get('is_sensitive')):
                        st.markdown("#### 🎯 敏感样本归因不稳定特征汇总")
                        trigger_feats = {}
                        for r in adv_batch['sensitive_samples']:
                            feat = r.get('trigger_feature')
                            if feat:
                                trigger_feats[feat] = trigger_feats.get(feat, 0) + 1

                        if trigger_feats:
                            trigger_df = pd.DataFrame([
                                {'特征': feat, '触发次数': count}
                                for feat, count in sorted(trigger_feats.items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(trigger_df, use_container_width=True)
                            st.info(f"💡 **分析提示**: 上述特征是最常导致解释敏感的不稳定特征。"
                                    f" 建议重点关注这些特征的数据质量和分布，考虑对其进行特征变换或正则化处理。")

            if adv_batch.get('sensitive_samples'):
                st.markdown("#### ⚠️ 敏感样本列表")
                sensitive_indices = [str(r['sample_idx']) for r in adv_batch['sensitive_samples']]
                st.info(f"样本索引: {', '.join(sensitive_indices)}")

    with tab4:
        st.header("📄 解释报告导出")
        st.caption("将所有分析结果汇总为一份交互式HTML报告")

        report_dir = st.text_input("报告输出目录", value="./output", key="report_dir")

        if st.button("📤 生成并导出HTML报告", type="primary", key="gen_report"):
            with st.spinner("正在生成HTML报告（可能需要几分钟）..."):
                try:
                    all_models = None
                    if pipeline.model_selector:
                        top_models_raw = pipeline.model_selector.get_top_models(n=3)
                        all_models = [(name, model) for name, model, _ in top_models_raw]

                    reporter = InterpretabilityReportExporter(
                        best_model, X, y, pipeline.task_type, best_model_name,
                        feature_names, pipeline.column_types, pipeline.random_state,
                        all_models=all_models,
                    )
                    import os
                    os.makedirs(report_dir, exist_ok=True)
                    report_path = os.path.join(report_dir, 'interpretability_report.html')
                    result = reporter.export_html_report(report_path)

                    if 'error' in result:
                        st.error(f"报告生成失败: {result['error']}")
                    else:
                        summary = result.get('summary', {})
                        st.success(f"✅ 报告已生成: {result.get('output_path', report_path)}")

                        st.markdown("---")
                        st.subheader("📋 报告摘要")
                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        col_r1.metric("模型", summary.get('model_name', 'Unknown'))
                        col_r2.metric("特征数", summary.get('n_features', 0))
                        col_r3.metric("样本数", f"{summary.get('n_samples', 0):,}")
                        col_r4.metric(
                            "可信度评级",
                            summary.get('credibility_grade', '未知'),
                        )

                        col_r5, col_r6 = st.columns(2)
                        col_r5.metric("整体一致性评分", f"{summary.get('overall_consistency', 0):.3f}")
                        col_r6.metric("对抗性检测通过率", f"{summary.get('adversarial_pass_rate', 0)*100:.1f}%")

                        grade = summary.get('credibility_grade', '')
                        grade_color = {'高': 'green', '中': 'orange', '低': 'red'}.get(grade, 'gray')
                        st.markdown(f"<h3 style='color: {grade_color};'>🏆 模型可信度评级: {grade}</h3>", unsafe_allow_html=True)
                        st.info(f"💡 {summary.get('suggestion', '')}")

                        if os.path.exists(report_path):
                            with open(report_path, 'r', encoding='utf-8') as f:
                                html_bytes = f.read().encode('utf-8')
                            st.download_button(
                                label="📥 下载HTML报告",
                                data=html_bytes,
                                file_name='interpretability_report.html',
                                mime='text/html',
                            )
                except Exception as e:
                    st.error(f"报告生成异常: {str(e)}")

        if st.session_state.interpretability_result:
            st.markdown("---")
            st.info("💡 已完成完整分析，可直接使用上方按钮导出报告")

    if st.button("➡️ 下一步：数据漂移检测", type="primary"):
        st.session_state.current_step = 6
        st.rerun()


def step_drift_detection():
    """步骤7: 数据漂移检测面板"""
    st.title("📊 数据漂移检测与告警")

    pipeline = st.session_state.pipeline

    mode_label = st.radio(
        "检测模式",
        options=["一次性全量检测", "滑动窗口持续监控"],
        format_func=lambda x: x,
        horizontal=True,
        key="drift_mode_radio",
        help="全量检测对比两份数据；滑动窗口模式按窗口/步长持续监控新数据流",
    )
    st.session_state.drift_detection_mode = (
        'once' if mode_label == "一次性全量检测" else 'sliding'
    )

    if st.session_state.drift_feature_weights is None:
        st.session_state.drift_feature_weights = _get_drift_feature_weights(pipeline)

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("🗂️ 数据集选择")

        st.markdown("**参考数据集 (Reference)")
        st.caption("用于对比的基准数据集，默认使用训练时的验证集")
        ref_option = st.radio(
            "选择参考数据来源",
            options=["使用Pipeline内部验证集", "上传CSV文件"],
            index=0,
            key="ref_data_source",
        )

        ref_df = None
        ref_name = ""

        if ref_option == "使用Pipeline内部验证集":
            if pipeline.diagnostician and hasattr(pipeline.diagnostician, "X_train") and pipeline.diagnostician.X_train is not None:
                ref_df = pipeline.diagnostician.X_train.copy()
                ref_name = "训练集(诊断划分)"
                st.success(f"✅ 已加载内部参考数据集，共 {len(ref_df)} 行 {len(ref_df.columns)} 列")
            elif pipeline.X_full is not None:
                ref_df = pipeline.X_full.copy()
                ref_name = "完整特征矩阵"
                st.success(f"✅ 已加载Pipeline特征矩阵，共 {len(ref_df)} 行 {len(ref_df.columns)} 列")
            else:
                st.warning("⚠️ Pipeline内部暂无数据，请先完成模型训练或上传CSV参考数据")
        else:
            ref_file = st.file_uploader(
                "上传参考数据集CSV",
                type=["csv"],
                key="ref_file_upload",
            )
            if ref_file is not None:
                try:
                    ref_df = pd.read_csv(ref_file)
                    ref_name = ref_file.name
                    st.session_state.drift_reference_df = ref_df
                    st.success(f"✅ 成功加载参考数据: {len(ref_df)} 行 x {len(ref_df.columns)} 列")
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")

        st.markdown("---")
        st.markdown("**待检测数据集 (New)")
        if st.session_state.drift_detection_mode == 'sliding':
            st.caption("滑动窗口模式：该数据作为持续流入的数据流，按步长切片进窗口")
        else:
            st.caption("部署后新进来的数据，需要检测是否发生分布偏移的数据")
        new_option = st.radio(
            "选择待检测数据来源",
            options=["使用Pipeline内部测试集", "上传CSV文件"],
            index=0,
            key="new_data_source",
        )

        new_df = None
        new_name = ""

        if new_option == "使用Pipeline内部测试集":
            if pipeline.diagnostician and hasattr(pipeline.diagnostician, "X_test") and pipeline.diagnostician.X_test is not None:
                new_df = pipeline.diagnostician.X_test.copy()
                new_name = "测试集(诊断划分)"
                st.success(f"✅ 已加载待检测数据，共 {len(new_df)} 行 {len(new_df.columns)} 列")
            else:
                st.warning("⚠️ Pipeline内部暂无测试集，请先完成诊断或上传CSV")
        else:
            new_file = st.file_uploader(
                "上传待检测数据集CSV",
                type=["csv"],
                key="new_file_upload",
            )
            if new_file is not None:
                try:
                    new_df = pd.read_csv(new_file)
                    new_name = new_file.name
                    st.session_state.drift_new_df = new_df
                    st.success(f"✅ 成功加载待检测数据: {len(new_df)} 行 x {len(new_df.columns)} 列")
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")

        st.markdown("---")

        storage_path = st.text_input(
            "告警记录存储路径",
            value="./drift_alerts.json",
            key="drift_storage_path",
        )

        if st.session_state.drift_detection_mode == 'sliding':
            run_drift_btn = _render_sliding_window_controls(
                pipeline, ref_df, new_df, ref_name, new_name, storage_path
            )
        else:
            col_run_l, col_run_r = st.columns([1, 1])
            with col_run_r:
                run_drift_btn = st.button("🔍 开始漂移检测", type="primary", key="run_drift_button")

    with right_col:
        alert_banner_displayed = False
        active_result = (
            st.session_state.drift_window_result
            if st.session_state.drift_detection_mode == 'sliding'
            else st.session_state.drift_detection_result
        )

        if active_result:
            result = active_result
            if 'error' in result:
                st.error(result['error'])
            else:
                alert_banner_displayed = True
                _render_drift_alert_banner(result)
                if result.get('data_insufficient'):
                    st.warning(f"⏳ {result.get('window_note', '数据不足，结果仅供参考')}")

        if (run_drift_btn or active_result):
            if run_drift_btn and st.session_state.drift_detection_mode == 'once':
                prep = _prepare_drift_dataframes(pipeline, ref_df, new_df)
                if prep is None:
                    return
                feature_col_types, ref_filtered, new_filtered = prep

                with st.spinner("正在进行数据漂移检测..."):
                    try:
                        detector = DriftDetector(
                            reference_data=ref_filtered,
                            column_types=feature_col_types,
                        )
                        result = detector.detect(new_filtered)

                        weighted_psi = compute_weighted_psi(
                            result['feature_psi'],
                            st.session_state.drift_feature_weights,
                        )

                        st.session_state.drift_trend_tracker.record(result, weighted_psi)
                        st.session_state.drift_weighted_psi = weighted_psi

                        try:
                            storage = AlertStorage(storage_path=storage_path)
                            storage.save_alert(
                                result,
                                dataset_name=f"{ref_name}_vs_{new_name}",
                            )
                        except Exception:
                            pass

                        st.session_state.drift_detection_result = result
                        st.session_state.drift_last_detector = detector
                        st.session_state.drift_last_ref_df = ref_filtered
                        st.session_state.drift_last_new_df = new_filtered

                        if not alert_banner_displayed:
                            _render_drift_alert_banner(result)

                    except Exception as e:
                        st.error(f"漂移检测失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        return

            result = active_result

            if result is None:
                return

            if 'error' in result:
                return

            if not alert_banner_displayed:
                _render_drift_alert_banner(result)
                if result.get('data_insufficient'):
                    st.warning(f"⏳ {result.get('window_note', '数据不足，结果仅供参考')}")

            _render_drift_metrics_overview(result, st.session_state.drift_weighted_psi)

            st.markdown("---")

            tab_heatmap, tab_distributions, tab_trend, tab_weighted = st.tabs([
                "🔥 单特征漂移热力图",
                "📈 分布对比图",
                "📊 漂移趋势追踪",
                "⚖️ 加权PSI对比",
            ])

            with tab_heatmap:
                _render_drift_heatmap(result)
                _render_drift_drilldown_panel()

            with tab_distributions:
                _render_drift_distribution_plots(result, pipeline)

            with tab_trend:
                _render_drift_trend_chart(
                    st.session_state.drift_trend_tracker,
                    result.get('n_total_features', 0),
                )

            with tab_weighted:
                _render_drift_weighted_psi(
                    result, st.session_state.drift_weighted_psi,
                    st.session_state.drift_feature_weights,
                )

            st.markdown("---")
            _render_drift_history(storage_path)

            st.markdown("---")
            export_c1, export_c2, export_c3 = st.columns([2, 1, 1])
            with export_c2:
                if st.button("📄 漂移报告导出", type="primary", key="drift_export_btn"):
                    _export_drift_report(result)
            with export_c3:
                if st.button("➡️ 下一步：导出Pipeline", type="primary", key="drift_next_btn"):
                    st.session_state.current_step = 7
                    st.rerun()

    if (st.session_state.drift_detection_mode == 'sliding'
            and st.session_state.drift_auto_monitoring
            and st.session_state.drift_monitor is not None):
        if st.session_state.drift_monitor.can_slide():
            time.sleep(int(st.session_state.drift_auto_interval))
            st.rerun()
        else:
            st.session_state.drift_auto_monitoring = False


def _get_drift_feature_weights(pipeline):
    """从Pipeline特征选择模块提取归一化的特征重要性权重"""
    analyzer = getattr(pipeline, 'feature_analyzer', None)
    if analyzer is None or getattr(analyzer, 'feature_names_', None) is None:
        return None
    try:
        importances = analyzer.get_all_importances()
    except Exception:
        return None
    if importances is None or len(importances) == 0:
        return None
    avail = [c for c in ['random_forest', 'permutation', 'l1_regularization']
             if c in importances.columns]
    if not avail:
        return None
    weights = importances[avail].mean(axis=1).fillna(0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.to_dict()


def _prepare_drift_dataframes(pipeline, ref_df, new_df):
    """准备漂移检测所需的特征列类型与过滤后的数据框"""
    if ref_df is None or new_df is None:
        st.error("❌ 请先选择参考数据集和待检测数据集!")
        return None

    feature_col_types = {
        col: col_type
        for col, col_type in pipeline.column_types.items()
        if col != pipeline.target_column
    }

    ref_feature_cols = [
        c for c in ref_df.columns
        if c in feature_col_types and feature_col_types[c] in ('numeric', 'categorical')
    ]

    ref_filtered = ref_df[ref_feature_cols].copy()
    new_feature_cols = [c for c in ref_feature_cols if c in new_df.columns]
    new_filtered = new_df[new_feature_cols].copy()

    if ref_filtered.empty:
        st.error("❌ 参考数据集中没有可分析的数值/分类特征，请检查列类型配置。")
        return None

    return feature_col_types, ref_filtered, new_filtered


def _render_sliding_window_controls(pipeline, ref_df, new_df, ref_name, new_name, storage_path):
    """渲染滑动窗口模式的控制面板，返回是否触发了首次检测"""
    st.markdown("**🪟 滑动窗口配置**")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        window_size = st.number_input(
            "窗口大小(行)", min_value=10, max_value=100000, value=1000, step=50,
            key="drift_window_size",
        )
    with col_w2:
        step_size = st.number_input(
            "步长(行)", min_value=1, max_value=10000, value=100, step=10,
            help="每滑动一次窗口前进的行数",
            key="drift_step_size",
        )

    monitor_cfg_key = (
        f"{ref_name}|{new_name}|"
        f"{len(ref_df) if ref_df is not None else 0}|"
        f"{len(new_df) if new_df is not None else 0}|"
        f"{window_size}|{step_size}"
    )
    monitor = st.session_state.drift_monitor

    if monitor is None or getattr(monitor, '_cfg_key', None) != monitor_cfg_key:
        if st.button("🚀 初始化监控器并加载数据流", type="primary", key="init_monitor_btn"):
            prep = _prepare_drift_dataframes(pipeline, ref_df, new_df)
            if prep is None:
                return False
            feature_col_types, ref_filtered, new_filtered = prep
            monitor = SlidingWindowDriftMonitor(
                reference_data=ref_filtered,
                column_types=feature_col_types,
                window_size=int(window_size),
                step_size=int(step_size),
            )
            monitor._cfg_key = monitor_cfg_key
            monitor.add_data(new_filtered)
            st.session_state.drift_monitor = monitor
            st.session_state.drift_window_result = None
            st.session_state.drift_weighted_psi = None
            st.success(f"✅ 监控器已初始化，已载入 {len(new_filtered)} 行数据流")
            st.rerun()
        return False

    status = monitor.get_status()
    st.caption(
        f"缓冲区: {status['buffer_size']} 行 | 当前窗口: "
        f"[{status['window_start']}:{status['window_start']+status['window_size']}] | "
        f"可滑动: {'是' if status['can_slide'] else '否'}"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        auto_interval = st.number_input(
            "自动检测间隔(秒)",
            min_value=5, max_value=3600, value=int(st.session_state.drift_auto_interval),
            step=5, key="drift_auto_interval_input",
        )
        st.session_state.drift_auto_interval = int(auto_interval)
    with col_b:
        auto_toggle = st.toggle(
            "自动监控", value=st.session_state.drift_auto_monitoring,
            key="drift_auto_toggle",
            help="开启后按间隔自动滑动并检测；关闭需等待当前周期结束",
        )
        st.session_state.drift_auto_monitoring = auto_toggle

    btn_detect = st.button("🔬 执行检测(当前窗口)", type="primary", key="detect_window_btn")
    btn_next = st.button("⏭️ 下一窗口", key="slide_window_btn",
                         disabled=not status['can_slide'])
    btn_reset = st.button("♻️ 重置监控器", key="reset_monitor_btn")

    if btn_reset:
        monitor.reset()
        st.session_state.drift_window_result = None
        st.session_state.drift_weighted_psi = None
        st.session_state.drift_auto_monitoring = False
        st.rerun()

    triggered = False

    if btn_detect:
        triggered = _run_window_detection(monitor, storage_path, ref_name, new_name)
    elif btn_next:
        if monitor.slide():
            triggered = _run_window_detection(monitor, storage_path, ref_name, new_name)
        else:
            st.warning("已到达数据流末尾，无法继续滑动")

    if st.session_state.drift_auto_monitoring:
        if monitor.can_slide():
            monitor.slide()
            _run_window_detection(monitor, storage_path, ref_name, new_name)
        else:
            st.session_state.drift_auto_monitoring = False
            st.info("数据流已全部检测完毕，自动监控已停止")

    return triggered


def _run_window_detection(monitor, storage_path, ref_name, new_name):
    """对当前窗口执行一次检测并记录趋势"""
    result = monitor.detect_current_window()
    if 'error' in result:
        st.error(result['error'])
        return False

    weighted_psi = compute_weighted_psi(
        result.get('feature_psi', {}),
        st.session_state.drift_feature_weights,
    )

    st.session_state.drift_trend_tracker.record(result, weighted_psi)
    st.session_state.drift_window_result = result
    st.session_state.drift_weighted_psi = weighted_psi
    st.session_state.drift_last_detector = monitor._detector
    st.session_state.drift_last_ref_df = monitor.reference_data
    st.session_state.drift_last_new_df = monitor.get_current_window()

    try:
        storage = AlertStorage(storage_path=storage_path)
        storage.save_alert(result, dataset_name=f"{ref_name}_vs_{new_name}")
    except Exception:
        pass

    return True


def _render_drift_trend_chart(tracker, n_total_features):
    """渲染漂移趋势追踪图：PSI折线 + 漂移特征数柱状 + 参考线 + 警戒区间阴影"""
    st.subheader("📊 漂移趋势追踪")
    st.caption("横轴为检测序号，左纵轴为PSI值(折线)，右纵轴为漂移特征数(柱状)")

    records = tracker.get_records()
    if not records:
        st.info("暂无趋势数据，请先执行检测（全量检测或滑动窗口检测均会记录）")
        return

    seqs = [r['seq'] for r in records]
    psis = [r['overall_psi'] for r in records]
    n_drifted = [r['n_drifted'] for r in records]
    retrain_line = n_total_features * 0.3

    streaks = tracker.get_warning_streaks()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for s in streaks:
        ax1.axvspan(s['start_idx'] - 0.4, s['end_idx'] + 0.4,
                    color='red', alpha=0.12, zorder=0)

    ax1.plot(seqs, psis, 'o-', color='#2980b9', linewidth=2,
             markersize=6, label='PSI值', zorder=3)
    ax1.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5,
                label='稳定线 PSI=0.1')
    ax1.axhline(y=0.25, color='orange', linestyle='--', linewidth=1.5,
                label='警戒线 PSI=0.25')
    ax1.set_xlabel('检测序号')
    ax1.set_ylabel('PSI值', color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#2980b9')

    ax2 = ax1.twinx()
    ax2.bar(seqs, n_drifted, alpha=0.3, color='#e74c3c', width=0.5,
            label='漂移特征数', zorder=1)
    if retrain_line > 0:
        ax2.axhline(y=retrain_line, color='red', linestyle='--', linewidth=1.5,
                    label=f'重训线 特征数={retrain_line:.1f}')
    ax2.set_ylabel('漂移特征数量', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax1.set_title('漂移趋势追踪')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if streaks:
        longest = max(streaks, key=lambda s: s['length'])
        st.error(
            f"🚨 连续 {longest['length']} 次超警戒，强烈建议重训"
            f"（区间: 第{longest['start_seq']}~{longest['end_seq']}次检测，图中红色阴影标注）"
        )
    else:
        max_streak = tracker.get_max_streak()
        if max_streak >= 1:
            st.info(f"当前连续超警戒次数: {max_streak}（达到3次将触发重训告警）")

    with st.expander("📋 查看趋势数据明细"):
        trend_df = pd.DataFrame(records)
        display_cols = [c for c in ['seq', 'timestamp', 'overall_psi', 'weighted_psi',
                                    'n_drifted', 'n_total_features', 'overall_alert_level']
                        if c in trend_df.columns]
        st.dataframe(trend_df[display_cols], use_container_width=True)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        if st.button("🗑️ 清空趋势记录", key="clear_trend_btn"):
            tracker.clear()
            st.rerun()
    with col_t2:
        st.caption(f"共记录 {len(records)} 次检测")


def _render_drift_weighted_psi(result, weighted_psi, feature_weights):
    """渲染加权PSI与非加权PSI对比"""
    st.subheader("⚖️ 特征重要性加权PSI")

    overall_psi = result.get('overall_psi', 0.0)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.metric("非加权PSI", f"{overall_psi:.4f}",
                  delta="所有特征一视同仁", delta_color="off")
    with col_w2:
        if weighted_psi is not None:
            st.metric("加权PSI", f"{weighted_psi:.4f}",
                      delta=f"差值: {weighted_psi - overall_psi:+.4f}")
        else:
            st.metric("加权PSI", "不可用")

    if weighted_psi is None:
        st.warning(
            "⚠️ 加权PSI不可用（需先完成特征选择）。"
            "请在「特征重要性评估」步骤运行后再回到本页，加权PSI将自动启用。"
        )
        return

    if not feature_weights:
        return

    feature_psi = result.get('feature_psi', {})
    rows = []
    for feat, psi_info in feature_psi.items():
        psi_val = psi_info.get('psi_value', 0.0) if isinstance(psi_info, dict) else 0.0
        if not np.isfinite(psi_val):
            continue
        w = feature_weights.get(feat, 0.0)
        rows.append({
            '特征': feat,
            'PSI值': round(float(psi_val), 4),
            '重要性权重': round(float(w), 4),
            '加权贡献': round(float(psi_val) * float(w), 4),
        })
    if rows:
        df_w = pd.DataFrame(rows).sort_values('加权贡献', ascending=False)
        st.markdown("**各特征加权贡献明细（按加权贡献降序）**")
        st.dataframe(df_w, use_container_width=True,
                     height=min(400, len(df_w) * 35 + 40))

    st.caption("💡 加权PSI = Σ(特征PSI × 归一化重要性权重)。对模型预测影响大的特征漂移将获得更高权重。")


def _render_drift_drilldown_panel():
    """渲染漂移根因下钻面板"""
    st.markdown("---")
    st.subheader("🔍 漂移根因下钻")

    detector = st.session_state.drift_last_detector
    new_df = st.session_state.drift_last_new_df

    if detector is None or new_df is None:
        st.info("请先执行漂移检测，然后选择漂移特征查看详情")
        return

    result = (st.session_state.drift_window_result
              if st.session_state.drift_detection_mode == 'sliding'
              else st.session_state.drift_detection_result)
    if result is None:
        st.info("请先执行漂移检测")
        return

    drifted = result.get('drifted_features', [])
    all_feats = list(result.get('feature_tests', {}).keys())

    if not all_feats:
        st.info("无可用特征")
        return

    options = (drifted if drifted else all_feats)
    default_idx = 0

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        selected = st.selectbox(
            "选择特征查看下钻详情",
            options=options,
            index=default_idx,
            key="drift_drilldown_feature",
            format_func=lambda x: f"{'🚨' if x in drifted else '○'} {x}",
        )
    with col_d2:
        st.caption(f"漂移特征 {len(drifted)} 个 | 可下钻特征 {len(all_feats)} 个")

    if selected is None:
        return

    try:
        drill = detector.get_feature_drilldown(selected, new_df)
    except Exception as e:
        st.error(f"下钻失败: {str(e)}")
        return

    if 'error' in drill:
        st.error(drill['error'])
        return

    with st.expander(f"📊 {selected} 详情面板", expanded=True):
        _render_drift_drilldown_detail(drill)


def _render_drift_drilldown_detail(drill):
    """渲染单个特征的下钻详情内容"""
    col_type = drill.get('type', '')

    if col_type == 'numeric':
        stats = drill.get('stats', {})
        ref_s = stats.get('reference', {})
        new_s = stats.get('new', {})

        st.markdown("**描述统计对比**")
        stat_rows = [
            ('均值', ref_s.get('mean', 0), new_s.get('mean', 0)),
            ('中位数', ref_s.get('median', 0), new_s.get('median', 0)),
            ('标准差', ref_s.get('std', 0), new_s.get('std', 0)),
            ('最小值', ref_s.get('min', 0), new_s.get('min', 0)),
            ('最大值', ref_s.get('max', 0), new_s.get('max', 0)),
            ('缺失率', ref_s.get('missing_rate', 0), new_s.get('missing_rate', 0)),
            ('有效样本数', ref_s.get('count', 0), new_s.get('count', 0)),
        ]
        stats_df = pd.DataFrame(stat_rows, columns=['指标', '参考集', '新数据'])
        stats_df['变化'] = stats_df['新数据'] - stats_df['参考集']
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("**分位数对比**")
        quantiles = drill.get('quantiles', [])
        if quantiles:
            q_df = pd.DataFrame(quantiles)
            st.dataframe(q_df, use_container_width=True, hide_index=True)

        kde = drill.get('kde', {})
        if kde.get('available'):
            st.markdown("**KDE密度曲线叠加**")
            fig, ax = plt.subplots(figsize=(10, 5))
            grid = kde['grid']
            ax.plot(grid, kde['ref_density'], label='参考集', color='#2980b9', linewidth=2)
            ax.plot(grid, kde['new_density'], label='新数据', color='#e67e22', linewidth=2)
            ax.fill_between(grid, kde['ref_density'], alpha=0.2, color='#2980b9')
            ax.fill_between(grid, kde['new_density'], alpha=0.2, color='#e67e22')
            ax.set_xlabel(drill.get('feature', ''))
            ax.set_ylabel('密度')
            ax.set_title(f"{drill.get('feature', '')} KDE密度曲线叠加")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"KDE不可用: {kde.get('reason', '样本数不足')}")

    elif col_type == 'categorical':
        stats = drill.get('stats', {})
        ref_s = stats.get('reference', {})
        new_s = stats.get('new', {})

        st.markdown("**描述统计对比**")
        cat_stat_rows = [
            ('类别数', ref_s.get('n_unique', 0), new_s.get('n_unique', 0)),
            ('众数', ref_s.get('top', ''), new_s.get('top', '')),
            ('众数频次', ref_s.get('top_freq', 0), new_s.get('top_freq', 0)),
            ('缺失率', ref_s.get('missing_rate', 0), new_s.get('missing_rate', 0)),
            ('有效样本数', ref_s.get('count', 0), new_s.get('count', 0)),
        ]
        cat_stats_df = pd.DataFrame(cat_stat_rows, columns=['指标', '参考集', '新数据'])
        st.dataframe(cat_stats_df, use_container_width=True, hide_index=True)

        waterfall = drill.get('waterfall', [])
        if waterfall:
            st.markdown("**类别占比变化瀑布图**（参考占比 → 新占比的增减）")
            top_n = min(len(waterfall), 15)
            wf = waterfall[:top_n]
            cats = [str(w['category'])[:15] for w in wf]

            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.8), 6))
            x = np.arange(len(wf))
            for i, w in enumerate(wf):
                ref_p = w['reference']
                delta = w['delta']
                ax.bar(i, ref_p, color='#95a5a6', alpha=0.5, width=0.6,
                       label='参考占比' if i == 0 else None)
                if delta >= 0:
                    ax.bar(i, delta, bottom=ref_p, color='#27ae60', alpha=0.85,
                           width=0.6, label='占比增加' if i == 0 else None)
                else:
                    ax.bar(i, abs(delta), bottom=ref_p + delta, color='#e74c3c',
                           alpha=0.85, width=0.6, label='占比减少' if i == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('占比')
            ax.set_title(f"{drill.get('feature', '')} 类别占比变化瀑布图")
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            wf_df = pd.DataFrame(wf)
            wf_df.columns = ['类别', '参考占比', '新占比', '变化']
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"特征类型 {col_type} 暂不支持下钻")





def _render_drift_alert_banner(result):
    """渲染顶部告警横幅"""
    alert_level = result.get('overall_alert_level', 'stable')
    banner_info = result.get('alert_banner', {})
    retraining = result.get('retraining_advice', {})

    alert_colors = {
        'stable': '#d5f5e3',
        'mild_drift': '#fdebd0',
        'severe_drift': '#fadbd8',
    }
    alert_icons = {
        'stable': '✅',
        'mild_drift': '⚠️',
        'severe_drift': '🚨',
    }
    alert_text_colors = {
        'stable': '#186a3b',
        'mild_drift': '#9c640c',
        'severe_drift': '#922b21',
    }

    color = alert_colors.get(alert_level, '#ecf0f1')
    icon = alert_icons.get(alert_level, 'ℹ️')
    text_color = alert_text_colors.get(alert_level, '#2c3e50')

    summary = banner_info.get('summary', '')
    drifted_features = banner_info.get('drifted_feature_details', [])
    action = retraining.get('action', 'continue_monitoring')
    reason = retraining.get('reason', '')
    urgency = retraining.get('urgency', 'low')

    urgency_badges = {
        'low': '<span style="background:#a9dfbf; color:#1e8449; padding:2px 8px; border-radius:10px; font-size:11px;">低</span>',
        'medium': '<span style="background:#fad7a0; color:#ca6f1e; padding:2px 8px; border-radius:10px; font-size:11px;">中</span>',
        'high': '<span style="background:#f5b7b1; color:#c0392b; padding:2px 8px; border-radius:10px; font-size:11px;">高</span>',
    }
    urgency_badge = urgency_badges.get(urgency, '')

    drifted_list_html = ''
    if drifted_features:
        items = ''.join([
            f"<li><strong>{d['feature']}</strong>: {d['direction_desc']}</li>"
            for d in drifted_features[:15]
        ])
        drifted_list_html = f"""
            <div style="margin-top: 12px;">
                <details style="cursor: pointer;">
                    <summary style="font-weight: 600;">🔍 漂移特征详情（{len(drifted_features)}个）
                    </summary>
                    <ul style="margin-top: 10px; padding-left: 20px;">
                        {items}
                    </ul>
                </details>
            </div>
        """

    html = f"""
        <div style="
            background: {color};
            border-left: 6px solid {text_color};
            padding: 18px 22px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="font-size: 20px; font-weight: bold; color: {text_color}; margin-bottom: 10px;">
                {icon} {banner_info.get('label', '')}
                &nbsp;&nbsp;{urgency_badge}
            </div>
            <div style="font-size: 15px; color: {text_color}; margin-bottom: 12px;">
                {summary}
            </div>
            <div style="background: rgba(255, 255, 255, 0.6);
                 padding: 12px 16px;
                 border-radius: 6px;
                 font-size: 14px;
                 color: {text_color};">
                <strong>💡 重训建议：</strong>
                <span style="margin-left: 8px;">
                    {reason}
                </span>
                <div style="margin-top: 6px;">
                    <strong>建议动作：</strong>
                    <span style="margin-left: 8px; font-weight: 600;">
                        {action}
                    </span>
                </div>
            </div>
            {drifted_list_html}
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_drift_metrics_overview(result, weighted_psi=None):
    """渲染整体指标卡片"""
    psi = result.get('overall_psi', 0.0)
    psi_grade = result.get('overall_psi_grade', 'stable')
    n_drifted = result.get('n_drifted', 0)
    n_total = result.get('n_total_features', 0)
    corrected_thresh = result.get('corrected_p_threshold', 0.05)
    original_thresh = result.get('original_p_threshold', 0.05)

    grade_cn = {
        'stable': '✅ 稳定',
        'mild_drift': '⚠️ 轻度漂移',
        'severe_drift': '🚨 严重漂移',
    }

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric(
            "整体 PSI 数值",
            f"{psi:.4f}",
            delta=grade_cn.get(psi_grade, psi_grade),
            delta_color='inverse' if psi_grade == 'severe_drift' else 'normal',
        )
    with col_m2:
        st.metric("漂移 / 总特征数", f"{n_drifted} / {n_total}")

    with col_m3:
        drift_pct = (n_drifted / n_total * 100) if n_total > 0 else 0
        st.metric("漂移特征占比", f"{drift_pct:.1f}%")

    with col_m4:
        if weighted_psi is not None:
            st.metric(
                "加权 PSI",
                f"{weighted_psi:.4f}",
                delta=f"差值 {weighted_psi - psi:+.4f}",
            )
        else:
            st.metric("加权 PSI", "不可用", delta="需先完成特征选择")

    with col_m5:
        st.metric(
            "Bonferroni 校正阈值",
            f"{corrected_thresh:.2e}",
            delta=f"原始: {original_thresh} ÷ {n_total} 特征",
        )


def _render_drift_heatmap(result):
    """渲染单特征漂移热力图"""
    st.subheader("🔥 单特征漂移热力图")
    st.caption("颜色越深表示漂移越显著，灰色表示未检测到显著漂移")

    feature_tests = result.get('feature_tests', {})
    feature_psi = result.get('feature_psi', {})

    if not feature_tests:
        st.info("暂无特征检验数据")
        return

    rows = []
    for feat, test in feature_tests.items():
        psi_info = feature_psi.get(feat, {})
        p_val = test.get('p_value', 1.0)
        stat = test.get('statistic', 0.0)
        is_drifted = test.get('is_drifted', False)
        psi_val = psi_info.get('psi_value', 0.0) if np.isfinite(psi_info.get('psi_value', 0.0)) else 0.0
        direction = test.get('direction', 'none')
        test_type = test.get('test', '')
        ftype = '数值型' if test_type == 'ks_2samp' else '分类型'

        if direction == 'mean_increase':
            dir_text = '均值上升'
        elif direction == 'mean_decrease':
            dir_text = '均值下降'
        elif 'category_' in direction:
            dir_text = '类别占比变化'
        else:
            dir_text = '无显著方向'

        rows.append({
            '特征名': feat,
            '类型': ftype,
            '检验方法': test_type,
            '统计量': round(stat, 4),
            'p值': f"{p_val:.3g}",
            'PSI值': round(psi_val, 4),
            '是否漂移': '是' if is_drifted else '否',
            '漂移方向': dir_text,
            '-log10(p值)': round(-np.log10(max(p_val, 1e-20)) if p_val > 0 else 20, 2),
        })

    df_heatmap = pd.DataFrame(rows)

    heatmap_cols = ['特征名', '类型', '检验方法', '统计量', 'p值', 'PSI值', '是否漂移', '漂移方向']
    display_df = df_heatmap[heatmap_cols].copy()

    display_df = display_df.sort_values(
        by=['是否漂移', 'PSI值'],
        ascending=[False, False],
    ).reset_index(drop=True)

    def highlight_rows(s):
        if s['是否漂移'] == '是':
            return ['background-color: #fadbd8'] * len(s)
        return [''] * len(s)

    styled = display_df.style.apply(highlight_rows, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(600, len(display_df) * 35 + 50),
    )

    st.markdown("---")

    st.caption("💡 说明：")
    st.markdown("- **p值** < Bonferroni校正阈值的特征被判定为显著漂移（标红高亮显示）")
    st.markdown("- **PSI值**: < 0.1 稳定，0.1~0.25 轻度漂移，>0.25 严重漂移")


def _render_drift_distribution_plots(result, pipeline):
    """渲染分布对比图"""
    st.subheader("📈 漂移特征分布对比")
    st.caption("参考分布(蓝色) vs 新数据分布(橙色) 直方图叠加对比")

    dist_data = result.get('distribution_data', {})
    drifted = result.get('drifted_features', [])

    if not dist_data:
        st.info("暂无分布数据")
        return

    features_to_show = drifted if drifted else list(dist_data.keys())

    if not features_to_show:
        st.info("无可绘图特征")
        return

    n_cols = 2
    n_features = min(len(features_to_show), 10)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 5 * n_rows),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feat in enumerate(features_to_show[:10]):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        data = dist_data.get(feat, {})
        ftype = data.get('type', 'numeric')

        if ftype == 'numeric':
            ref_hist = data.get('reference', {}).get('hist', np.array([]))
            ref_edges = data.get('reference', {}).get('edges', np.array([]))
            new_hist = data.get('new', {}).get('hist', np.array([]))
            new_edges = data.get('new', {}).get('edges', np.array([]))

            if len(ref_hist) > 0 and len(ref_edges) > 1:
                width_ref = np.diff(ref_edges)
                width_new = np.diff(new_edges) if len(new_edges) > 1 else width_ref

                ax.bar(
                    ref_edges[:-1],
                    ref_hist,
                    width=width_ref,
                    alpha=0.5,
                    label='参考分布',
                    color='#3498db',
                    edgecolor='white',
                )
                new_offset = (width_new[0] / 2) if len(new_hist) > 0 else 0
                ax.bar(
                    new_edges[:-1] + new_offset,
                    new_hist,
                    width=width_new,
                    alpha=0.5,
                    label='新分布',
                    color='#e67e22',
                    edgecolor='white',
                )

            ref_mean = data.get('reference', {}).get('mean', 0)
            new_mean = data.get('new', {}).get('mean', 0)
            ax.axvline(
                ref_mean,
                color='#2980b9',
                linestyle='--',
                linewidth=2,
                label=f'参考均值: {ref_mean:.3f}',
            )
            ax.axvline(
                new_mean,
                color='#d35400',
                linestyle='--',
                linewidth=2,
                label=f'新均值: {new_mean:.3f}',
            )
            ax.legend(fontsize=8)
            ax.set_xlabel(feat)
            ax.set_ylabel('密度')
            ax.grid(True, alpha=0.3)

        else:
            cats = data.get('reference', {}).get('categories', [])
            ref_props = data.get('reference', {}).get('proportions', [])
            new_props = data.get('new', {}).get('proportions', [])

            if cats:
                x = np.arange(len(cats))
                width = 0.35
                ax.bar(
                    x - width / 2,
                    ref_props,
                    width=width,
                    alpha=0.7,
                    label='参考分布',
                    color='#3498db',
                )
                ax.bar(
                    x + width / 2,
                    new_props,
                    width=width,
                    alpha=0.7,
                    label='新分布',
                    color='#e67e22',
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(c)[:12] for c in cats],
                    rotation=45,
                    ha='right',
                    fontsize=8,
                )
                ax.legend(fontsize=8)
                ax.set_xlabel(feat)
                ax.set_ylabel('占比')
                ax.grid(True, alpha=0.3, axis='y')

    for idx in range(len(features_to_show[:10]), n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_drift_history(storage_path: str):
    """渲染历史告警记录"""
    st.subheader("📜 历史告警记录")

    try:
        storage = AlertStorage(storage_path)
        history = storage.get_all_alerts(limit=50)

        if not history:
            st.info("暂无历史告警记录")
            return

        history_df = pd.DataFrame(history)

        level_map = {
            'stable': '✅ 稳定',
            'mild_drift': '⚠️ 轻度',
            'severe_drift': '🚨 严重',
        }
        urgency_map = {
            'low': '低',
            'medium': '中',
            'high': '高',
        }
        action_map = {
            'continue_monitoring': '继续监控',
            'monitor_closely': '密切监控',
            'consider_retrain': '考虑重训',
            'retrain_immediately': '立即重训',
        }

        display_cols = [
            'timestamp', 'dataset_name',
            'overall_alert_level',
            'overall_psi',
            'n_drifted',
            'n_total_features',
            'retraining_action',
            'retraining_urgency',
        ]

        if all(c in history_df.columns for c in display_cols):
            history_disp = history_df[display_cols].copy()
            history_disp.columns = [
                '检测时间',
                '数据集',
                '告警级别',
                'PSI',
                '漂移特征数',
                '总特征数',
                '建议动作',
                '紧急度',
            ]
            history_disp['告警级别'] = history_disp['告警级别'].map(level_map).fillna(history_disp['告警级别'])
            history_disp['建议动作'] = history_disp['建议动作'].map(action_map).fillna(history_disp['建议动作'])
            history_disp['紧急度'] = history_disp['紧急度'].map(urgency_map).fillna(history_disp['紧急度'])
            history_disp['PSI'] = history_disp['PSI'].round(4)

            st.dataframe(
                history_disp,
                use_container_width=True,
                height=min(300, len(history_disp) * 35 + 50),
            )

    except Exception as e:
        st.info(f"读取历史记录暂不可用: {str(e)}")


def _export_drift_report(result):
    """导出漂移检测报告"""
    try:
        exporter = DriftReportExporter(result)

        html_content = exporter.export_html()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"drift_report_{timestamp}.html"

        st.success(f"✅ 报告已生成: {filename}")

        st.download_button(
            label="💾 下载 HTML 报告",
            data=html_content,
            file_name=filename,
            mime="text/html",
            key="download_drift_report",
        )

    except Exception as e:
        st.error(f"报告生成失败: {str(e)}")


def step_export():
    """步骤8: Pipeline导出"""
    st.title("📦 Pipeline导出")
    st.markdown("导出完整的可复用Pipeline，支持多种格式。")

    output_dir = st.text_input("导出目录", value="./output")
    include_onnx = st.checkbox("包含ONNX格式导出（如果支持）", value=False)

    if st.button("📤 导出Pipeline", type="primary"):
        with st.spinner("正在导出Pipeline..."):
            pipeline = st.session_state.pipeline

            try:
                results = pipeline.export_pipeline(
                    output_dir=output_dir,
                    include_onnx=include_onnx,
                )

                st.success("✅ Pipeline导出成功！")

                st.markdown("---")
                st.subheader("📁 导出文件")

                for key, path in results.items():
                    st.text(f"  • {key}: {path}")

                st.markdown("---")
                st.subheader("📋 模型卡预览")

                card_path = results.get('model_card', '')
                if card_path and os.path.exists(card_path):
                    with open(card_path, 'r', encoding='utf-8') as f:
                        card_content = f.read()
                    st.markdown(card_content)

            except Exception as e:
                st.error(f"导出失败: {str(e)}")

    st.markdown("---")
    st.info("💡 导出的文件包含：")
    st.text("  • pipeline.pkl / pipeline.joblib - 完整Pipeline模型文件")
    st.text("  • model.onnx (可选) - ONNX格式模型")
    st.text("  • prediction_api.py - 预测API脚本模板")
    st.text("  • model_card.md - 模型卡文档")
    st.text("  • feature_stats.json - 训练数据特征统计（用于漂移检测）")
    st.text("  • drift_detector.py - 数据漂移检测脚本")


def main():
    """主函数"""
    init_session_state()
    step_navigation()

    steps = [
        step_data_upload,
        step_feature_engineering,
        step_feature_selection,
        step_model_selection,
        step_model_diagnosis,
        step_interpretability,
        step_drift_detection,
        step_export,
    ]

    current_step = st.session_state.current_step
    if 0 <= current_step < len(steps):
        steps[current_step]()


if __name__ == "__main__":
    main()
