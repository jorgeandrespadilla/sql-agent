import os

from langchain_community.utilities.sql_database import SQLDatabase

from sql_agent.config import AppConfig, ConfigProvider

config_provider = ConfigProvider(".env")
app_config = AppConfig(config_provider)

os.environ["OPENAI_API_KEY"] = app_config.OPENAI_API_KEY

db = SQLDatabase.from_uri("sqlite:///data/chinook.db")


def check_db_connection():
    db.run("SELECT 1;")
    # print("Database dialect: ", db.dialect)
    # print("Sample data:")
    # print(db.get_usable_table_names())
    # print(db.run("SELECT * FROM Artist LIMIT 10;"))


if __name__ == "__main__":
    try:
        check_db_connection()
    except Exception as e:
        print("Error: ", e)
        print("Database connection failed.")
        exit(1)
    print("Database connection successful.")


