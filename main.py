import ast
import os
import re
from typing import Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
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
from langchain_core.tools import BaseTool

from sql_agent.config import AppConfig, ConfigProvider
from sql_agent.examples import SQL_EXAMPLES
from sql_agent.prompts import SQL_SYSTEM_PREFIX
import streamlit as st

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


def query_as_list(db: SQLDatabase, query: str):
    """
    Run the given query on the database and return the results as a list.
    """
    res = str(db.run(query))
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [string.strip() for string in res]  # Remove extra spaces
    # res = [re.sub(r"\b\d+\b", "", string).strip() for string in res] # Remove numbers and extra spaces
    return list(set(res))


def build_retriever_tool(db: SQLDatabase):
    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""  # TODO: Improve description
    description = """Use to look up values to filter on for:
    - Artist Name
    - Album Title
    Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    return retriever_tool


def run_agent(
    user_input: str,
    db: SQLDatabase,
    extra_tools: Sequence[BaseTool] = [],
    use_examples: bool = False
):
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
        extra_tools=extra_tools,
        agent_type="openai-tools",
        verbose=True
    )
    agent_input = {
        "input": user_input
    } if use_examples else user_input
    result = agent.invoke(agent_input)  # type: ignore
    return result


def generate_response(query: str):
    extra_tools = [build_retriever_tool(db)]
    result = run_agent(
        query,
        db,
        extra_tools=extra_tools,
        use_examples=True
    )
    return result


def run_test():
    # test_full_prompt()
    # print(query_as_list(db, "SELECT Name FROM Artist"))
    # exit()

    # query = "What are the names of all the artists in the database?"
    # query = "How many albums does alis chein have?"
    query = "How many albums does the twelve of berlin have?"
    result = generate_response(query)
    print("Question: ", query)
    print("Answer:")
    print(result["output"])


def run_app():
    st.title('🤖 SQL Agent')

    with st.form('my_form'):
        query = st.text_area('Enter query:', 'How many albums does the twelve of berlin have?')
        submitted = st.form_submit_button('Ask')
        if submitted:
            response = generate_response(query)["output"]
            st.info(response)


if __name__ == "__main__":
    try:
        check_db_connection()
    except Exception as e:
        print("Error: ", e)
        print("Database connection failed.")
        exit(1)
    print("Database connection successful.")

    # run_test()
    run_app()
