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
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False


def step_navigation():
    """步骤导航侧边栏"""
    st.sidebar.title("🔬 AutoML Pipeline")
    st.sidebar.markdown("---")

    steps = [
        "📁 数据上传与探索",
        "⚙️ 自动特征工程",
        "🎯 特征重要性评估",
        "🤖 自动模型选择",
        "📊 模型对比与诊断",
        "📦 Pipeline导出",
    ]

    for i, step in enumerate(steps):
        if i == st.session_state.current_step:
            st.sidebar.markdown(f"**➡️ {step}**")
        elif i < st.session_state.current_step:
            st.sidebar.markdown(f"✅ {step}")
        else:
            st.sidebar.markdown(f"⬜ {step}")

    st.sidebar.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("⬅️ 上一步", disabled=st.session_state.current_step == 0):
            st.session_state.current_step = max(0, st.session_state.current_step - 1)
            st.rerun()
    with col2:
        if st.button("下一步 ➡️", disabled=st.session_state.current_step >= 5):
            st.session_state.current_step = min(5, st.session_state.current_step + 1)
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

    st.subheader("📊 数据概况报告")

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

        if st.button("➡️ 下一步：导出Pipeline", type="primary"):
            st.session_state.current_step = 5
            st.rerun()


def step_export():
    """步骤6: Pipeline导出"""
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
        step_export,
    ]

    current_step = st.session_state.current_step
    if 0 <= current_step < len(steps):
        steps[current_step]()


if __name__ == "__main__":
    main()
