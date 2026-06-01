import pathlib
from typing import Protocol, Sequence

from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader

import config


class TemplateConfig:
    """
    A simple class to hold necessary information for generating a SQL query
    from a Jinja2 template.

    """

    def __init__(self, template_name: str, view_name: str, source_table: str) -> None:
        self.template_name = template_name
        self.view_name = view_name
        self.source_table = source_table


class TemplateConfigFactory:
    """
    Factory for creating TemplateConfig objects based on combinations of
    the datasets used and number of versions. The main function
    get_template_info() returns a list of TemplateConfig objects that can be
    used by TemplatedViewQuery to generate SQL queries.

    """

    def __init__(
        self,
        datasets: Sequence[str],
        versions: int,
        template_name: str,
        view_name: str,
        source_table: str,
    ) -> None:
        self.template_info_list: list[TemplateConfig] | None = None
        self.datasets = datasets
        self.versions = versions
        self.template_name = template_name
        self.view_name = view_name
        self.source_table = source_table

    def get_template_info(self) -> list[TemplateConfig]:
        """
        Return a list of TemplateConfig objects. If the list has not yet
        been created, it will be created first and then returned.

        """
        if self.template_info_list is None:
            self.template_info_list = self._make_template_info_list()

        return self.template_info_list

    def _make_template_info_list(self) -> list[TemplateConfig]:
        """
        Construct a list of TemplateConfig objects.

        """
        self.template_info_list = []
        for dataset in self.datasets:
            for version in range(1, self.versions + 1):
                self.template_info_list.append(
                    TemplateConfig(
                        template_name=self.template_name.format(version=version),
                        view_name=self.view_name.format(
                            dataset=dataset, version=version
                        ),
                        source_table=self.source_table.format(dataset=dataset),
                    )
                )

        return self.template_info_list


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
    TEMPLATE_CONFIGS = [
        TemplateConfigFactory(
            datasets=['train', 'test'],
            versions=1,
            template_name='01_initial_cleaning_template.sql.j2',
            view_name='{dataset}_cleaned',
            source_table='{dataset}_raw',
        ),
        TemplateConfigFactory(
            datasets=['train', 'test'],
            versions=2,
            template_name='02_features_v{version}_template.sql.j2',
            view_name='{dataset}_features_v{version}',
            source_table='{dataset}_cleaned',
        ),
    ]

    view_templates = [
        TemplatedViewQuery(
            env=JINJA_ENV,
            template=SQL_DIR.joinpath(info.template_name),
            view_name=f'airline_data.{info.view_name}',
            source_table=f'airline_data.{info.source_table}',
        )
        for config in TEMPLATE_CONFIGS
        for info in config.get_template_info()
    ]

    execute_queries(view_templates)
