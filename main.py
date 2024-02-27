import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from sql_agent.config import AppConfig, ConfigProvider
from sql_agent.examples import SQL_EXAMPLES
from sql_agent.prompts import SQL_SYSTEM_PREFIX


# Configure app
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


def build_full_prompt():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        SQL_EXAMPLES,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=SQL_SYSTEM_PREFIX,
        suffix="",
    )
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return full_prompt


def test_full_prompt():
    full_prompt = build_full_prompt()
    prompt_val = full_prompt.invoke({
        "input": "How many arists are there",
        "top_k": 5,
        "dialect": "SQLite",
        "agent_scratchpad": [],
    })
    print(prompt_val.to_string())


def run_agent(user_input: str, use_examples: bool = False):
    """
    Run the agent with the given query.

    When use_examples is True, the agent will use a few-shot prompt to generate the query, in other words, it will include examples in the prompt before running the query.
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = build_full_prompt() if use_examples else None
    agent = create_sql_agent(
        llm,
        db=db,
        prompt=prompt,
        agent_type="openai-tools",
        verbose=True
    )
    agent_input = {
        "input": user_input
    } if use_examples else user_input
    result = agent.invoke(agent_input)  # type: ignore
    return result


if __name__ == "__main__":
    try:
        check_db_connection()
    except Exception as e:
        print("Error: ", e)
        print("Database connection failed.")
        exit(1)
    print("Database connection successful.")

    # test_full_prompt()

    result = run_agent("What are the names of all the artists in the database?", use_examples=True)
    print(result)
