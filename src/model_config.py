import pathlib
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from src.db_config import ProjectStage, DatasetSplit, build_table_name


class OptimizationProfile:
    """
    Store optimization parameters for an optuna study.

    """

    N_TRIALS: int = 10
    SEED: int = 256
    SCORING: str = 'roc_auc'


class HyperparamProfile(Enum):
    """
    Store the hyperparameter search space for an optuna study.

    """

    WIDE = {
        'n_estimators': {'type': 'int', 'bounds': (50, 300)},
        'max_depth': {'type': 'int', 'bounds': (3, 10)},
        'learning_rate': {
            'type': 'float',
            'bounds': (0.01, 0.2),
            'kwargs': {'log': True},
        },
        'num_leaves': {'type': 'int', 'bounds': (20, 150)},
    }
    RESTRICTED = {
        'n_estimators': {'type': 'int', 'bounds': (50, 300)},
        'max_depth': {'type': 'int', 'bounds': (3, 6)},
        'learning_rate': {
            'type': 'float',
            'bounds': (0.01, 0.2),
            'kwargs': {'log': True},
        },
        'num_leaves': {'type': 'int', 'bounds': (20, 50)},
        'min_child_samples': {'type': 'int', 'bounds': (20, 50)},
    }


@dataclass
class FeatureProfile:
    """
    Store the features that will be dropped or added for a model version.
    These are separated into numeric, categorical, and passthrough features
    for processing through a ColumnTransformer.

    """

    drop_cols: list[str] = field(default_factory=list)
    add_num_cols: list[str] = field(default_factory=list)
    add_cat_cols: list[str] = field(default_factory=list)
    passthrough_cols: list[str] = field(default_factory=list)


class ModelVersion(Enum):
    """
    Store model versions and the corresponding information regarding the
    data they will use.

    """

    V1_0 = ('v1.0', ProjectStage.FEATURE_ENG, 1, HyperparamProfile.WIDE)
    V1_1 = ('v1.1', ProjectStage.FEATURE_ENG, 1, HyperparamProfile.RESTRICTED)
    V2_0 = ('v2.0', ProjectStage.FEATURE_ENG, 2, HyperparamProfile.RESTRICTED)
    V2_1 = ('v2.1', ProjectStage.FEATURE_ENG, 3, HyperparamProfile.RESTRICTED)
    V3_0 = ('v3.0', ProjectStage.FEATURE_ENG, 4, HyperparamProfile.RESTRICTED)

    def __init__(
        self,
        version_str: str,
        stage: ProjectStage,
        sql_version: int,
        hyperparam_profile: HyperparamProfile,
    ) -> None:
        self.version_str = version_str
        self.stage = stage
        self.sql_version = sql_version
        self.hyperparam_profile = hyperparam_profile

    @property
    def feature_profile(self) -> FeatureProfile:
        """
        Get the feature profile for a model version.

        """
        return _FEATURE_REGISTRY[self]

    def get_table_name(self, dataset: DatasetSplit) -> str:
        """
        Get the BigQuery view name for this model version and a given dataset.

        """
        return build_table_name(self.stage, dataset, version=self.sql_version)


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the data dtypes according to the DTYPE_MAP. Only applies to columns
    that are present in both the DTYPE_MAP and the dataframe.

    """
    active_dtypes = {
        col: dtype for (col, dtype) in DTYPE_MAP.items() if col in df.columns
    }
    return df.astype(active_dtypes)


MODELS_DIR = pathlib.Path(__file__).parents[1].joinpath('models')
DTYPE_MAP = {
    'id': 'int64',
    'gender': 'str',
    'is_loyal_customer': 'int64',
    'age': 'int64',
    'is_personal_travel': 'int64',
    'class': 'str',
    'flight_distance': 'int64',
    'wifi_service': 'int64',
    'convenient_time': 'int64',
    'online_booking_ease': 'int64',
    'gate_location': 'int64',
    'food_drink': 'int64',
    'online_boarding': 'int64',
    'seat_comfort': 'int64',
    'inflight_entertainment': 'int64',
    'onboard_service': 'int64',
    'leg_room': 'int64',
    'baggage_handling': 'int64',
    'checkin_service': 'int64',
    'inflight_service': 'int64',
    'cleanliness': 'int64',
    'departure_delay': 'int64',
    'arrival_delay': 'int64',
    # Engineered features ---------
    'is_business_class': 'int64',
    'cabin_rating': 'float64',
    'service_rating': 'float64',
    'satisfied': 'int64',
}
TARGET = 'satisfied'
NUMERIC_COLS = ['age', 'flight_distance', 'departure_delay', 'arrival_delay']
CATEGORICAL_COLS = ['gender', 'is_loyal_customer', 'is_personal_travel', 'class']

# These columns are all ratings from 0-5 or 1-5
RATED_COLS = [
    'wifi_service',
    'convenient_time',
    'online_booking_ease',
    'gate_location',
    'food_drink',
    'online_boarding',
    'seat_comfort',
    'inflight_entertainment',
    'onboard_service',
    'leg_room',
    'baggage_handling',
    'checkin_service',
    'inflight_service',
    'cleanliness',
]

_FEATURE_REGISTRY = {
    ModelVersion.V1_0: FeatureProfile(
        drop_cols=['gender'],
        passthrough_cols=['is_loyal_customer', 'is_personal_travel'],
    ),
    ModelVersion.V1_1: FeatureProfile(
        drop_cols=['gender'],
        passthrough_cols=['is_loyal_customer', 'is_personal_travel'],
    ),
    ModelVersion.V2_0: FeatureProfile(
        drop_cols=[
            'age',
            'class',
            'departure_delay',
            'flight_distance',
            'food_drink',
            'gate_location',
            'gender',
            'online_booking_ease',
        ],
        passthrough_cols=[
            'is_loyal_customer',
            'is_personal_travel',
            'is_business_class',
        ],
    ),
    ModelVersion.V2_1: FeatureProfile(
        drop_cols=[
            'age',
            'class',
            'baggage_handling',
            'checkin_service',
            'cleanliness',
            'convenient_time',
            'departure_delay',
            'flight_distance',
            'food_drink',
            'gate_location',
            'gender',
            'online_booking_ease',
            'inflight_entertainment',
            'inflight_service',
            'leg_room',
            'onboard_service',
            'seat_comfort',
        ],
        add_num_cols=['cabin_rating', 'service_rating'],
        passthrough_cols=[
            'is_loyal_customer',
            'is_personal_travel',
            'is_business_class',
        ],
    ),
    ModelVersion.V3_0: FeatureProfile(
        drop_cols=[
            'age',
            'arrival_delay',
            'baggage_handling',
            'checkin_service',
            'class',
            'cleanliness',
            'convenient_time',
            'departure_delay',
            'flight_distance',
            'food_drink',
            'gate_location',
            'gender',
            'inflight_entertainment',
            'inflight_service',
            'leg_room',
            'onboard_service',
            'online_booking_ease',
            'seat_comfort',
        ],
        add_num_cols=['cabin_rating', 'service_rating'],
        passthrough_cols=[
            'is_business_class',
            'is_loyal_customer',
            'is_personal_travel',
        ],
    ),
}
