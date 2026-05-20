import pathlib
from typing import Protocol, Sequence

from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader

import config


class SQLQueryProvider(Protocol):
    def get_query(self) -> str: ...


class TemplatedViewQuery(SQLQueryProvider):
    """
    SQL query provider that generates a view creation query based on a Jinja2
    template. The Jinja2 environment, path to the template file, name of the
    view to be created, and source table used in the view must be provided.

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
        template_file = self.template.name
        jinja_template = self.env.get_template(template_file)

        return jinja_template.render(
            view_name=self.view_name, source_table=self.source_table
        )


def execute_queries(query_providers: Sequence[SQLQueryProvider]) -> None:
    """
    Execute queries to BigQuery by running the SQL files contained within
    each query provider.

    """
    client = bigquery.Client(project=config.PROJECT_ID)

    for provider in query_providers:
        query = provider.get_query()
        query_job = client.query(query)
        query_job.result()  # wait for the view to be created


if __name__ == '__main__':
    SQL_DIR = pathlib.Path(__file__).parents[1].joinpath('sql')
    JINJA_ENV = Environment(loader=FileSystemLoader(SQL_DIR))

    view_template_01 = SQL_DIR.joinpath('01_initial_cleaning_template.sql.j2')

    view_templates_01 = [
        TemplatedViewQuery(
            env=JINJA_ENV,
            template=view_template_01,
            view_name='airline_data.train_cleaned',
            source_table='airline_data.train_raw',
        ),
        TemplatedViewQuery(
            env=JINJA_ENV,
            template=view_template_01,
            view_name='airline_data.test_cleaned',
            source_table='airline_data.test_raw',
        ),
    ]

    execute_queries(view_templates_01)
