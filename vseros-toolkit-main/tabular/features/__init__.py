# ml_foundry/features/__init__.py

"""Unified exports for all tabular feature generators."""

from .base import FeatureGenerator, FitStrategy

from .aggregation import AggregationGenerator, RollingAggregationGenerator
from .datetime import DatetimeFeatureGenerator, DateDifferenceGenerator
from .interaction import (
    NumericalInteractionGenerator,
    CategoricalInteractionGenerator,
    NumCatInteractionGenerator,
)

from .numerical.scaling import (
    StandardScalerGenerator,
    MinMaxScalerGenerator,
    RobustScalerGenerator,
)
from .numerical.transformations import (
    LogTransformer,
    SqrtTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
)
from .numerical.flags import ValueIndicator, IsNullIndicator, OutlierIndicator
from .numerical.binning import EqualWidthBinner, QuantileBinner, DecisionTreeBinner

from .categorical.ordinal import OrdinalEncoderGenerator
from .categorical.target_based import TargetEncoderGenerator, WoEEncoderGenerator
from .categorical.nominal import (
    OneHotEncoderGenerator,
    CountFrequencyEncoderGenerator,
    HashingEncoderGenerator,
)
from .categorical.combination import RareCategoryCombiner

from .text.bow import CountVectorizerGenerator, TfidfVectorizerGenerator
from .text.embeddings import PretrainedEmbeddingGenerator
from .text.statistics import TextStatisticsGenerator
from .text.transformer import TransformerEmbeddingGenerator

from .advanced.neighbors import NearestNeighborsFeatureGenerator
from .advanced.model_based import (
    KMeansFeatureGenerator,
    PCAGenerator,
    TreeLeafFeatureGenerator,
)
from .advanced.matrix_factorization import MatrixFactorizationFeatureGenerator
from .advanced.autoencoder import AutoencoderFeatureGenerator

__all__ = [
    "FeatureGenerator",
    "FitStrategy",
    "AggregationGenerator",
    "RollingAggregationGenerator",
    "DatetimeFeatureGenerator",
    "DateDifferenceGenerator",
    "NumericalInteractionGenerator",
    "CategoricalInteractionGenerator",
    "NumCatInteractionGenerator",
    "StandardScalerGenerator",
    "MinMaxScalerGenerator",
    "RobustScalerGenerator",
    "LogTransformer",
    "SqrtTransformer",
    "BoxCoxTransformer",
    "YeoJohnsonTransformer",
    "ValueIndicator",
    "IsNullIndicator",
    "OutlierIndicator",
    "EqualWidthBinner",
    "QuantileBinner",
    "DecisionTreeBinner",
    "OrdinalEncoderGenerator",
    "TargetEncoderGenerator",
    "WoEEncoderGenerator",
    "OneHotEncoderGenerator",
    "CountFrequencyEncoderGenerator",
    "HashingEncoderGenerator",
    "RareCategoryCombiner",
    "CountVectorizerGenerator",
    "TfidfVectorizerGenerator",
    "PretrainedEmbeddingGenerator",
    "TextStatisticsGenerator",
    "TransformerEmbeddingGenerator",
    "NearestNeighborsFeatureGenerator",
    "KMeansFeatureGenerator",
    "PCAGenerator",
    "TreeLeafFeatureGenerator",
    "MatrixFactorizationFeatureGenerator",
    "AutoencoderFeatureGenerator",
]