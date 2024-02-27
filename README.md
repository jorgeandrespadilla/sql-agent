# SQL Agent

This is a simple SQL Agent that can be used to run SQL queries against a database using LLMs. The main advantages of using the SQL Agent are:
- It can answer questions based on the databases’ schema as well as on the databases’ content (like describing a specific table).
- It can recover from errors by running a generated query, catching the traceback and regenerating it correctly.
- It can query the database as many times as needed to answer the user question.
- It will save tokens by only retrieving the schema from relevant tables.

## Setup

> Before you start, you need to install `poetry`.

1. Clone the repository
2. Install the dependencies: `poetry install`
3. Activate the virtual environment: `poetry shell`
4. Configure the environment variables in the `.env` file (use the `.env.example` as a template)
5. Initialize the database: `python -m scripts.init_db`
