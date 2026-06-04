import pathlib
from itertools import chain
from functools import partial
from typing import Iterator, Protocol, Sequence

from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader

import db_config


class SQLQueryProvider(Protocol):
    def get_query(self) -> str: ...

    def message(self) -> str: ...


class TemplatedViewQuery(SQLQueryProvider):
    """
    SQL query provider that generates a view creation query based on a Jinja2
    template.

    """

    def __init__(
        self,
        env: Environment,
        template: pathlib.Path,
        view_name: str,
        source_table: str,
    ) -> None:
        self.env = env
        self.template = template
        self.view_name = view_name
        self.source_table = source_table

    def get_query(self) -> str:
        """
        Render a Jinja2 template and return the query string.

        """
        template_file = self.template.name
        jinja_template = self.env.get_template(template_file)

        return jinja_template.render(
            view_name=self.view_name, source_table=self.source_table
        )

    def message(self) -> str:
        """
        Print a message noting the current query being sent.

        """
        template_parent = self.template.parent.name
        template_name = self.template.stem.split('.')[0]

        return f'Executing: {template_parent}/{template_name} -> view: {self.view_name}'


def generate_templates(
    env: Environment,
    datasets: Sequence[str],
    versions: int,
    template: pathlib.Path,
    view_name: str,
    source_table: str,
    dataset_location: str,
) -> Iterator[TemplatedViewQuery]:
    """
    Generator function that yields TemplatedViewQuery objects by iterating
    through datasets and versions.

    """
    if versions > 1:
        t_check = template.with_name(template.name.format(version=1)) == template
        v_check = view_name.format(dataset=datasets[0], version=1) == view_name.format(
            dataset=datasets[0], version=2
        )

        if t_check or v_check:
            raise ValueError(
                f"Multiple versions were given ({versions}), but at least one of"
                " `template` or `view_name` was not version formatted:\n"
                f'\t template: {template.name}\n\tview_name: {view_name}'
            )

    for dataset in datasets:
        for version in range(1, versions + 1):
            template_fmt = template.with_name(template.name.format(version=version))
            view_name_fmt = view_name.format(dataset=dataset, version=version)
            source_table_fmt = source_table.format(dataset=dataset)

            yield TemplatedViewQuery(
                env=env,
                template=template_fmt,
                view_name=f'{dataset_location}.{view_name_fmt}',
                source_table=f'{dataset_location}.{source_table_fmt}',
            )


def execute_queries(
    query_providers: Sequence[SQLQueryProvider], verbose: bool = True
) -> None:
    """
    Execute queries to BigQuery by running the SQL files contained within
    each query provider.

    """
    client = bigquery.Client(project=db_config.PROJECT_ID)

    for provider in query_providers:
        if verbose:
            print(provider.message())

        query = provider.get_query()
        query_job = client.query(query)
        query_job.result()  # wait for the view to be created


if __name__ == '__main__':
    SQL_DIR = pathlib.Path(__file__).parents[1].joinpath('sql')
    JINJA_ENV = Environment(loader=FileSystemLoader(SQL_DIR))
    STAGES = db_config.ProjectStage
    TEMPLATES = db_config.SQL_TEMPLATES
    TABLE_NAMES = db_config.SQL_TABLE_NAMES
    VERSIONS = db_config.SQL_TEMPLATE_VERSIONS
    SPLITS = db_config.DATASET_SPLITS

    make_templates = partial(
        generate_templates, env=JINJA_ENV, dataset_location=db_config.DATASET_LOCATION
    )

    # Generate the SQL queries used to make new views in BigQuery
    view_generators = [
        make_templates(
            datasets=SPLITS[STAGES.CLEANING],
            versions=VERSIONS[STAGES.CLEANING],
            template=SQL_DIR.joinpath(TEMPLATES[STAGES.CLEANING]),
            view_name=TABLE_NAMES[STAGES.CLEANING],
            source_table=TABLE_NAMES[STAGES.RAW],
        ),
        make_templates(
            datasets=SPLITS[STAGES.FEATURE_ENG],
            versions=VERSIONS[STAGES.FEATURE_ENG],
            template=SQL_DIR.joinpath(TEMPLATES[STAGES.FEATURE_ENG]),
            view_name=TABLE_NAMES[STAGES.FEATURE_ENG],
            source_table=TABLE_NAMES[STAGES.CLEANING],
        ),
    ]

    view_templates = list(chain.from_iterable(view_generators))
    execute_queries(view_templates)
