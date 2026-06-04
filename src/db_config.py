from enum import Enum


class ProjectStage(Enum):
    """
    Create a general structure for project stages.

    """

    RAW = 'raw'
    CLEANING = 'cleaning'
    FEATURE_ENG = 'feature_eng'


class DatasetSplit(Enum):
    """
    List the current available options for dataset splits.

    """

    TRAIN = 'train'
    TEST = 'test'


def build_table_name(
    stage: ProjectStage, dataset: DatasetSplit, version: int = 1
) -> str:
    """
    Build the BigQuery view name for a given project stage and dataset split.

    """
    template = SQL_TABLE_NAMES[stage]
    view_name = template.format(dataset=dataset.value, version=version)
    return f'{DATASET_LOCATION}.{view_name}'


PROJECT_ID = 'project-e18491ab-d50e-4c36-a7d'
DATASET_LOCATION = 'airline_data'  # location in BigQuery of all the data/views
SQL_TEMPLATES = {
    ProjectStage.CLEANING: '01_initial_cleaning_template.sql.j2',
    ProjectStage.FEATURE_ENG: '02_features_v{version}_template.sql.j2',
}
SQL_TABLE_NAMES = {
    ProjectStage.RAW: '{dataset}_raw',
    ProjectStage.CLEANING: '{dataset}_cleaned',
    ProjectStage.FEATURE_ENG: '{dataset}_features_v{version}',
}
SQL_TEMPLATE_VERSIONS = {
    ProjectStage.CLEANING: 1,
    ProjectStage.FEATURE_ENG: 4,
}
DATASET_SPLITS = {
    ProjectStage.RAW: [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value],
    ProjectStage.CLEANING: [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value],
    ProjectStage.FEATURE_ENG: [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value],
}
