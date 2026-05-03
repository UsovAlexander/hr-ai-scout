import os
from airflow.hooks.base import BaseHook
from clickhouse_driver import Client


class ClickHouseHook(BaseHook):
    """Хук для подключения к ClickHouse через clickhouse-driver."""

    conn_name_attr = 'clickhouse_conn_id'
    default_conn_name = 'clickhouse_default'
    conn_type = 'generic'
    hook_name = 'ClickHouse'

    def __init__(self, clickhouse_conn_id: str = default_conn_name):
        super().__init__()
        self.clickhouse_conn_id = clickhouse_conn_id
        self._client = None

    def get_conn(self) -> Client:
        if self._client is None:
            try:
                conn = self.get_connection(self.clickhouse_conn_id)
                self._client = Client(
                    host=conn.host or os.getenv('CLICKHOUSE_HOST', 'localhost'),
                    port=int(conn.port or os.getenv('CLICKHOUSE_PORT', 9000)),
                    user=conn.login or os.getenv('CLICKHOUSE_USER', 'default'),
                    password=conn.password or os.getenv('CLICKHOUSE_PASSWORD', ''),
                    database=conn.schema or os.getenv('CLICKHOUSE_DATABASE', 'default'),
                    settings={'use_numpy': True},
                )
            except Exception:
                # Fallback: читаем напрямую из переменных окружения
                self._client = Client(
                    host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
                    port=int(os.getenv('CLICKHOUSE_PORT', 9000)),
                    user=os.getenv('CLICKHOUSE_USER', 'default'),
                    password=os.getenv('CLICKHOUSE_PASSWORD', ''),
                    database=os.getenv('CLICKHOUSE_DATABASE', 'default'),
                    settings={'use_numpy': True},
                )
        return self._client

    def execute(self, query: str, params=None):
        return self.get_conn().execute(query, params or [])

    def query_dataframe(self, query: str):
        return self.get_conn().query_dataframe(query)

    def insert_dataframe(self, query: str, df):
        return self.get_conn().insert_dataframe(query, df)
