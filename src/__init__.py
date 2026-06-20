"""AutoML Pipeline 包"""

__version__ = '1.0.0'

__all__ = [
    'DataTypeInference',
    'DataExplorer',
    'DataSampler',
    'TargetValidator',
    'load_csv',
    'AutoFeatureEngineer',
    'LogTransformer',
    'PolynomialCrossTransformer',
    'MissingIndicatorTransformer',
    'BinningTransformer',
    'FrequencyEncoder',
    'TargetEncoder',
    'DateFeatureExtractor',
    'TfidfTextTransformer',
    'VarianceFilter',
    'HighCorrelationFilter',
    'FeatureImportanceAnalyzer',
    'IntersectionFeatureSelector',
    'AutoModelSelector',
    'ModelDefinition',
    'ModelDiagnostician',
    'compare_models',
    'format_model_comparison',
    'PipelineExporter',
    'LocalInterpreter',
    'GlobalInterpreter',
    'AdversarialExplainer',
    'InterpretabilityReportExporter',
    'ModelInterpretabilityAnalyzer',
    'AutoMLPipeline',
]


def __getattr__(name):
    """延迟导入，避免可选依赖缺失时导入失败"""
    if name in [
        'DataTypeInference', 'DataExplorer', 'DataSampler',
        'TargetValidator', 'load_csv'
    ]:
        from . import data_exploration
        return getattr(data_exploration, name)

    if name in [
        'AutoFeatureEngineer', 'LogTransformer',
        'PolynomialCrossTransformer', 'MissingIndicatorTransformer',
        'BinningTransformer', 'FrequencyEncoder', 'TargetEncoder',
        'DateFeatureExtractor', 'TfidfTextTransformer',
        'VarianceFilter', 'HighCorrelationFilter',
    ]:
        from . import feature_engineering
        return getattr(feature_engineering, name)

    if name in ['FeatureImportanceAnalyzer', 'IntersectionFeatureSelector']:
        from . import feature_selection
        return getattr(feature_selection, name)

    if name in ['AutoModelSelector', 'ModelDefinition']:
        from . import model_selector
        return getattr(model_selector, name)

    if name in ['ModelDiagnostician', 'compare_models', 'format_model_comparison']:
        from . import model_diagnosis
        return getattr(model_diagnosis, name)

    if name == 'PipelineExporter':
        from . import pipeline_exporter
        return getattr(pipeline_exporter, name)

    if name in [
        'LocalInterpreter', 'GlobalInterpreter',
        'AdversarialExplainer', 'InterpretabilityReportExporter',
        'ModelInterpretabilityAnalyzer',
    ]:
        from . import interpretability
        return getattr(interpretability, name)

    if name == 'AutoMLPipeline':
        from . import automl_pipeline
        return getattr(automl_pipeline, name)

    raise AttributeError(f"module 'src' has no attribute '{name}'")
