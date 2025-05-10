"""
Database-based prompt provider implementation.

This module provides an implementation of the PromptProvider interface that retrieves prompts
from various database systems using any of the supported database clients. The DBPromptProvider
can be used with a wide range of database systems, both relational (SQL) and non-relational (NoSQL),
making it versatile for different storage backends.

Supported database systems include:

Relational Databases:
- SQLite (sqlite3, aiosqlite)
- PostgreSQL (psycopg2, psycopg, asyncpg, aiopg)
- MySQL/MariaDB (mysql-connector-python, PyMySQL, aiomysql, asyncmy, mariadb)
- MS SQL Server (pymssql, pyodbc, aioodbc)
- Oracle (cx_Oracle, oracledb)

NoSQL Databases:
- MongoDB (pymongo, motor)
- Redis (redis, aioredis)
- Cassandra (cassandra-driver)
- CouchBase (couchbase)
- RethinkDB (rethinkdb)
- InfluxDB (influxdb-client)

The provider adapts to each database's specific API and provides a uniform interface
for retrieving prompts, handling both synchronous and asynchronous database clients.

Examples:
    # SQLite example
    import sqlite3

    # Create a database connection
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()

    # Set up a table for prompts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
        name TEXT PRIMARY KEY,
        template TEXT NOT NULL
    )
    ''')

    # Insert some sample data
    cursor.execute("INSERT OR REPLACE INTO prompts VALUES (?, ?)",
                  ('greeting', 'Hello, {{name}}! Welcome to our service.'))
    conn.commit()

    # Create a database prompt provider
    from agentle.prompts.prompt_providers.db_prompt_provider import DBPromptProvider

    db_provider = DBPromptProvider(
        connection=conn,
        query="SELECT template FROM prompts WHERE name = ?",
        param_conversion=lambda name: (name,)
    )

    # Load a prompt from the database
    prompt = db_provider.provide("greeting")

    # Compile the prompt with context data
    result = prompt.compile({"name": "Alice"})
    # Output: "Hello, Alice! Welcome to our service."


    # PostgreSQL example (with asyncpg)
    import asyncpg
    import asyncio

    async def setup_postgres_example():
        # Create a connection pool
        pool = await asyncpg.create_pool(
            user='postgres',
            password='password',
            database='prompts_db',
            host='localhost'
        )

        # Create a provider that works with asyncpg
        async_provider = DBPromptProvider(
            connection=pool,
            query="SELECT template FROM prompt_templates WHERE name = $1",
        )

        # Retrieve and use a prompt
        prompt = async_provider.provide("welcome_email")
        return prompt.content

    # MongoDB example
    from pymongo import MongoClient

    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')

    # Create a MongoDB provider
    mongo_provider = DBPromptProvider(
        connection=client,
        query="prompts_db.templates",  # Format: "database.collection"
        param_conversion=lambda name: {"name": name}
    )

    # Get a prompt from MongoDB
    prompt = mongo_provider.provide("password_reset")
"""

from __future__ import annotations
import asyncio
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Protocol,
    TYPE_CHECKING,
    Union,
    override,
    cast,
)

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider

# Import types only for type checking, without runtime dependency
if TYPE_CHECKING:
    import aiomysql
    import aiopg
    import aiosqlite
    import asyncmy
    import asyncpg
    import mariadb
    import motor.motor_asyncio
    import mysql.connector
    import oracledb
    import psycopg
    import psycopg2
    import pymongo
    import pymssql
    import pyodbc
    import redis
    import redis.asyncio as aioredis
    import sqlite3
    from aioodbc import Connection as AioOdbcConnection
    from cassandra.cluster import Session as CassandraSession
    from couchbase.cluster import Cluster as CouchbaseCluster
    from influxdb_client.client.influxdb_client import InfluxDBClient
    from rethinkdb import RethinkDB

    # Type for any supported database connection
    DBConnection = Union[
        # SQL databases
        sqlite3.Connection,
        aiosqlite.Connection,
        psycopg2.extensions.connection,
        psycopg.Connection,
        asyncpg.Connection,
        aiopg.Connection,
        mysql.connector.MySQLConnection,
        aiomysql.Connection,
        asyncmy.Connection,
        mariadb.Connection,
        pymssql.Connection,
        pyodbc.Connection,
        AioOdbcConnection,
        oracledb.Connection,
        # NoSQL databases
        pymongo.MongoClient[Any],
        motor.motor_asyncio.AsyncIOMotorClient[Any],
        redis.Redis,
        aioredis.Redis,
        CassandraSession,
        RethinkDB,
        InfluxDBClient,
        CouchbaseCluster,
    ]


class _AsyncQueryExecutor(Protocol):
    """Protocol for executing async database queries."""

    async def execute_query(self, query: str, params: Any) -> str:
        """Execute a database query and return the result as a string."""
        ...


class DBPromptProvider(PromptProvider):
    """
    A prompt provider that retrieves prompts from a database.

    This provider supports various database systems and can be configured
    with a custom query and parameter conversion function. It automatically
    detects the type of database connection provided and uses the appropriate
    query mechanism.

    The provider simplifies integration with existing database systems by providing
    a consistent interface regardless of the underlying database technology. It handles
    both synchronous and asynchronous connections and operations appropriately.

    Attributes:
        connection: The database connection object
        query: The SQL query or equivalent to execute
        param_conversion: A function to convert the prompt_id to query parameters

    Examples:
        # Using with SQLite
        import sqlite3
        conn = sqlite3.connect('prompts.db')

        provider = DBPromptProvider(
            connection=conn,
            query="SELECT content FROM prompts WHERE id = ?",
        )

        # Using with PostgreSQL (psycopg2)
        import psycopg2
        conn = psycopg2.connect("dbname=mydb user=postgres password=secret")

        provider = DBPromptProvider(
            connection=conn,
            query="SELECT content FROM prompts WHERE id = %s",
        )

        # Using with MySQL
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost",
            user="user",
            password="password",
            database="prompts_db"
        )

        provider = DBPromptProvider(
            connection=conn,
            query="SELECT content FROM prompts WHERE id = %s",
        )

        # Using with MongoDB
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')

        provider = DBPromptProvider(
            connection=client,
            query="prompts.templates",  # database.collection
            param_conversion=lambda prompt_id: {"name": prompt_id}
        )

        # Using with Redis
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)

        provider = DBPromptProvider(
            connection=r,
            query="",  # Query is ignored for Redis
            param_conversion=lambda prompt_id: f"prompt:{prompt_id}"
        )

        # Retrieving a prompt (works the same for all providers)
        prompt = provider.provide("greeting")
        compiled = prompt.compile({"name": "User"})
    """

    def __init__(
        self,
        connection: Any,
        query: str,
        param_conversion: Callable[[str], Any] = lambda x: (x,),
    ) -> None:
        """
        Initialize a database prompt provider.

        Args:
            connection: Any supported database connection
            query: The query to execute for retrieving prompts
            param_conversion: A function to convert the prompt_id to query parameters
        """
        super().__init__()
        self.connection = connection
        self.query = query
        self.param_conversion = param_conversion
        self._executor = self._get_executor()

    def _get_executor(self) -> _AsyncQueryExecutor:
        """
        Determine the appropriate executor based on the connection type.

        Returns:
            An executor object that implements the _AsyncQueryExecutor protocol
        """
        conn_type = type(self.connection).__name__

        # SQLite
        if conn_type in ("Connection", "sqlite3.Connection"):
            return SQLiteExecutor(self.connection)
        elif "aiosqlite" in conn_type:
            return AioSQLiteExecutor(self.connection)

        # PostgreSQL
        elif any(
            t in conn_type for t in ("psycopg2", "psycopg2.extensions.connection")
        ):
            return Psycopg2Executor(self.connection)
        elif "psycopg" in conn_type:
            return PsycopgExecutor(self.connection)
        elif "asyncpg" in conn_type:
            return AsyncpgExecutor(self.connection)
        elif "aiopg" in conn_type:
            return AiopgExecutor(self.connection)

        # MySQL/MariaDB
        elif any(t in conn_type for t in ("MySQLConnection", "mysql.connector")):
            return MySQLConnectorExecutor(self.connection)
        elif "pymysql" in conn_type or "PyMySQL" in conn_type:
            return PyMySQLExecutor(self.connection)
        elif "aiomysql" in conn_type:
            return AioMySQLExecutor(self.connection)
        elif "asyncmy" in conn_type:
            return AsyncMyExecutor(self.connection)
        elif "mariadb" in conn_type:
            return MariaDBExecutor(self.connection)

        # MS SQL
        elif "pymssql" in conn_type:
            return PyMSSQLExecutor(self.connection)
        elif "pyodbc" in conn_type:
            return PyODBCExecutor(self.connection)
        elif "aioodbc" in conn_type:
            return AioODBCExecutor(self.connection)

        # Oracle
        elif any(t in conn_type for t in ("cx_Oracle", "oracledb")):
            return OracleExecutor(self.connection)

        # MongoDB
        elif "MongoClient" in conn_type:
            return MongoExecutor(self.connection)
        elif "AsyncIOMotorClient" in conn_type:
            return MotorExecutor(self.connection)

        # Redis
        elif "Redis" in conn_type:
            return RedisExecutor(self.connection)

        # Cassandra
        elif "Session" in conn_type and hasattr(self.connection, "execute"):
            return CassandraExecutor(self.connection)

        # Other NoSQL
        elif "RethinkDB" in conn_type:
            return RethinkDBExecutor(self.connection)
        elif "InfluxDBClient" in conn_type:
            return InfluxDBExecutor(self.connection)
        elif (
            "Cluster" in conn_type
            and hasattr(self.connection, "query")
            and hasattr(self.connection, "bucket")
        ):
            return CouchbaseExecutor(self.connection)

        raise ValueError(f"Unsupported database connection type: {conn_type}")

    @override
    def provide(self, prompt_id: str) -> Prompt:
        """
        Retrieve a prompt from the database.

        Args:
            prompt_id: The identifier for the prompt to retrieve

        Returns:
            A Prompt object containing the content from the database

        Raises:
            Various database-specific exceptions may be raised
        """
        # Use asyncio.run for the async implementation
        try:
            # For environments where asyncio is already running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(
                    self.provide_async(prompt_id), loop
                ).result()
            else:
                return asyncio.run(self.provide_async(prompt_id))
        except RuntimeError:
            # If no event loop is available, create a new one
            return asyncio.run(self.provide_async(prompt_id))

    async def provide_async(self, prompt_id: str) -> Prompt:
        """
        Retrieve a prompt asynchronously from the database.

        Args:
            prompt_id: The identifier for the prompt to retrieve

        Returns:
            A Prompt object containing the content from the database
        """
        params = self.param_conversion(prompt_id)
        content = await self._executor.execute_query(self.query, params)
        return Prompt(content=content)


# Executor implementations for different database types


class SQLiteExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class AioSQLiteExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.execute(query, params) as cursor:
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class Psycopg2Executor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class PsycopgExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class AsyncpgExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        if isinstance(params, tuple):
            result = await self.conn.fetchrow(query, *params)
        else:
            result = await self.conn.fetchrow(query, params)
        if result is None:
            raise ValueError("No result found for query")
        return str(result[0])


class AiopgExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class MySQLConnectorExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class PyMySQLExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class AioMySQLExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class AsyncMyExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class MariaDBExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class PyMSSQLExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class PyODBCExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class AioODBCExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])


class OracleExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result is None:
                raise ValueError("No result found for query")
            return str(result[0])
        finally:
            cursor.close()


class MongoExecutor:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def execute_query(self, query: str, params: Sequence[Any]) -> str:
        # For MongoDB, query is a string representation of a query dict and collection
        # Format: "database.collection"
        db_name, collection_name = query.split(".")
        collection = self.client[db_name][collection_name]

        # Params should be a dict for MongoDB queries
        if isinstance(params, tuple) and len(params) > 0:
            filter_dict = {"name": params[0]}
        else:
            filter_dict = cast(dict[str, Any], params)

        result = collection.find_one(filter_dict)
        if result is None:
            raise ValueError(f"No document found with query: {filter_dict}")

        # Return the template field or the first field if template doesn't exist
        return str(
            result.get("template")
            or result.get("content")
            or next(iter(result.values()))
        )


class MotorExecutor:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def execute_query(self, query: str, params: Sequence[Any]) -> str:
        # For MongoDB, query is a string representation of a query dict and collection
        # Format: "database.collection"
        db_name, collection_name = query.split(".")
        collection = self.client[db_name][collection_name]

        # Params should be a dict for MongoDB queries
        if isinstance(params, tuple) and len(params) > 0:
            filter_dict = {"name": params[0]}
        else:
            filter_dict = cast(dict[str, Any], params)

        result = await collection.find_one(filter_dict)
        if result is None:
            raise ValueError(f"No document found with query: {filter_dict}")

        # Return the template field or the first field if template doesn't exist
        return str(
            result.get("template")
            or result.get("content")
            or next(iter(result.values()))
        )


class RedisExecutor:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def execute_query(self, query: str, params: Any) -> str:
        # For Redis, query is ignored, and params is used as the key
        key = params[0] if isinstance(params, tuple) else params

        # Handle both sync and async Redis clients
        if hasattr(self.conn, "get") and callable(self.conn.get):
            if asyncio.iscoroutinefunction(self.conn.get):
                result = await self.conn.get(key)
            else:
                result = self.conn.get(key)
        else:
            raise ValueError("Redis connection doesn't have a get method")

        if result is None:
            raise ValueError(f"No value found for key: {key}")

        # Decode if result is bytes
        if isinstance(result, bytes):
            return result.decode("utf-8")
        return str(result)


class CassandraExecutor:
    def __init__(self, session: Any) -> None:
        self.session = session

    async def execute_query(self, query: str, params: Any) -> str:
        # For Cassandra, we use the query directly
        if asyncio.iscoroutinefunction(self.session.execute):
            rows = await self.session.execute(query, params)
        else:
            rows = self.session.execute(query, params)

        if not rows:
            raise ValueError("No result found for query")

        # Get the first column of the first row
        return str(rows[0][0])


class RethinkDBExecutor:
    def __init__(self, conn: Any) -> None:
        self.r = conn

    async def execute_query(self, query: str, params: Any) -> str:
        # For RethinkDB, query is a table name, and params is used as the filter
        db_name, table_name = query.split(".")

        key = params[0] if isinstance(params, tuple) else params
        result = self.r.db(db_name).table(table_name).get(key).run()

        if result is None:
            raise ValueError(f"No document found with key: {key}")

        # Return the template field or the first field if template doesn't exist
        return str(
            result.get("template")
            or result.get("content")
            or next(iter(result.values()))
        )


class InfluxDBExecutor:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def execute_query(self, query: str, params: Sequence[Any]) -> str:
        # For InfluxDB, query is a flux query
        # We substitute params into the query as a simple approach
        if isinstance(params, tuple):
            for i, param in enumerate(params):
                query = query.replace(f"${i + 1}", str(param))

        query_api = self.client.query_api()
        result = query_api.query(query)

        if not result or not result[0].records:
            raise ValueError("No result found for query")

        # Get the value from the first record
        return str(result[0].records[0].values["_value"])


class CouchbaseExecutor:
    def __init__(self, cluster: Any) -> None:
        self.cluster = cluster

    async def execute_query(self, query: str, params: Any) -> str:
        # For Couchbase, query can be a bucket and key or an N1QL query
        if "SELECT" in query.upper():
            # N1QL query
            if asyncio.iscoroutinefunction(self.cluster.query):
                result = await self.cluster.query(
                    query, *params if isinstance(params, tuple) else params
                )
            else:
                result = self.cluster.query(
                    query, *params if isinstance(params, tuple) else params
                )

            if not result.rows():
                raise ValueError("No result found for query")

            row = result.rows()[0]
            return str(
                row.get("template") or row.get("content") or next(iter(row.values()))
            )
        else:
            # Key-value operation (bucket.collection format)
            bucket_name, collection_name = query.split(".")
            bucket = self.cluster.bucket(bucket_name)
            collection = bucket.collection(collection_name)

            key = params[0] if isinstance(params, tuple) else params

            if asyncio.iscoroutinefunction(collection.get):
                result = await collection.get(key)
            else:
                result = collection.get(key)

            if result is None:
                raise ValueError(f"No document found with key: {key}")

            content = result.content
            return str(
                content.get("template")
                or content.get("content")
                or next(iter(content.values()))
            )
