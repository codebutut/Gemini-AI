import pandas as pd
from sqlalchemy import create_engine, inspect, text
from typing import List, Dict, Any, Optional
from gemini_agent.utils.logger import get_logger
from gemini_agent.core.tools import tool

logger = get_logger(__name__)

@tool
def query_database(connection_string: str, query: str) -> str:
    """
    Executes a SQL query against a database and returns the results as a formatted string.
    
    Args:
        connection_string: SQLAlchemy connection string (e.g., 'sqlite:///example.db').
        query: The SQL query to execute.
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            df = pd.read_sql_query(text(query), connection)
            return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return f"Error: {e}"

@tool
def list_database_tables(connection_string: str) -> List[str]:
    """
    Lists all tables in the specified database.
    """
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        return [f"Error: {e}"]

@tool
def get_database_schema(connection_string: str, table_name: str) -> str:
    """
    Returns the schema (columns and types) for a specific table.
    """
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        schema = [f"{col['name']} ({col['type']})" for col in columns]
        return ", ".join(schema)
    except Exception as e:
        logger.error(f"Failed to get schema for {table_name}: {e}")
        return f"Error: {e}"
