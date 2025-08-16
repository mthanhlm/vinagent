# Banking and Finance Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent-docs/blob/main/docs/tutorials/guides/banking_agent.ipynb)

In today's fast-paced financial world, the ability to quickly extract insights from data can make or flaw business decisions. Imagine having an AI assistant that can instantly answer questions like "Which customer has the highest deposit balance?" to take care of or "Show me the total balance of all loans, deposits today" to measure the growth rate. You can issue business policies or regulations just in day or in real-time. It is feasible by using Vinagent to automatically drive SQL engine to obtain the trust and worthy answers. Therefore, your organization will gain long-term benefits compared to those without ultilize AI Agent.

## Why banking needs Intelligent Agent

The financial industry generates massive amounts of data every second. Traditional reporting systems often fall short because they:

- Require technical expertise: Business users need IT teams to write SQL queries
- Lack real-time insights: Static reports become outdated quickly
- Miss critical patterns: Human analysts can't create a swift SQL query at scale.
- Slow decision-making: Waiting for reports delays crucial business decisions

Real-world examples demonstrate the power of automated reporting systems:

JPMorgan Chase invested heavily in upgrading their financial reporting systems after the 2008 crisis. Their new system provides daily insights into capital positions, liquidity, and risk, enabling them to optimize capital allocation dynamically and resume share repurchases faster than competitors.

DBS Bank Singapore transformed their reporting infrastructure with a centralized data lake and analytics tools, reducing month-end closing times from 12 days to just 3 days. This improvement dramatically enhanced operational efficiency and drove revenue growth.

These success stories highlight a crucial truth: timely, automated reporting isn't just convenient, it's a competitive advantage.

That is why vinagent supports a strong Banking and Finance agent that works on on-premise and cloud environments. In this notebook, let's study how to create a Banking and Finance agent to optimize the decision making in terms of speed and accuracy relying on SQL engine. This is a list of features you will study:

- Create a agent to question and answering on any business query.
- Integrate special SQL tools for database analysis.
- Drive an end-to-end AI agent workflow to transform, execute, and illustrate SQL table.
- Create a cycling SQL workflow to optimize quality of generated SQL query over many circles.

## Text-2-SQL banking and finance Agent

Our banking agent will act as an intelligent intermediary between business users and database systems. It can convert natural language questions into SQL queries and then execute queries safely with built-in validation queries tool before triggering Moreover, it can present results in human-readable format and handle complex multi-table joins automatically. With a context-awareness of database, the agent can provide a high precise result from simple to complex queries.

To finish any business query, This is agent workflow should be sequentially executed:

1. Understand the question: Detect and extract the main user intent from natural language.
2. Explore database structure: Identify relevant tables and relationships to find the right tables shoule be used to answer the question.
3. Generate SQL query: Create syntactically correct SQL relying on how they understand database schema and business query.
4. Validate query: Check for common mistakes and security issues before execution. This is to early prevent SQL execution errors that delays the system.
5. Execute and format: Run query and present results clearly ensuring human understanding.

## Prequisite Installation

Before we dive into building, let's set up the necessary tools:

```python
%pip install vinagent langchain_openai==0.3.7
```

You'll also need an [OpenAI key](https://platform.openai.com/api-keys) for the LLM models.


```python
# %%writefile .env
# OPENAI_API_KEY=your_api_key
```

## Initialize LLM

Let's initialize LLM model as a brain of AI Agent. In this tutorial, we select `GPT-4o-mini` as baseline model. You can refer to [Live-bench](https://livebench.ai/#/) leaderboard to select the best model for coding task.


```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))

llm = ChatOpenAI(
    model = "o4-mini"
)
```

## Initialize SQL Database

For this tutorial, we'll use a realistic banking database with six interconnected tables that mirror real-world financial institutions:

**List of tables**

1. Customer

    - CustomerID (PK): Unique identifier for each customer.
    - Name: Full name of the customer.
    - Address: Address of the customer.
    - Contact: Contact details (e.g., phone number, email).
    - Username (Unique): Username used to log in.
    - Password: Encrypted password for authentication.

2. Account

    - AccountID (PK): Unique identifier of the account.
    - CustomerID (FK → Customer.CustomerID): Identifies the owner of the account.
    - ProductType: Type of account (e.g., savings, current).
    - ProductCategory: Category (e.g., retail, business).
    - Balance: Current balance of the account.

3. Transactions

    - TransactionID (PK, Auto Increment): Unique identifier of the transaction.
    - AccountID (FK → Account.AccountID): Account involved in the transaction.
    - Type: Transaction type (e.g., deposit, withdrawal, transfer).
    - Amount: Transaction amount.
    - Timestamp: Date and time of the transaction.

4. Deposit

    - DepositID (PK): Unique identifier of the deposit product.
    - CustomerID (FK → Customer.CustomerID): Customer who owns the deposit.
    - ProductType: Type of deposit (e.g., fixed, recurring).
    - ProductCategory: Category of deposit product.
    - Balance: Current deposit balance.
    - Term: Deposit term (e.g., 12 months).

5. Loan

    - LoanID (PK): Unique identifier of the loan product.
    - CustomerID (FK → Customer.CustomerID): Customer who took the loan.
    - ProductType: Type of loan (e.g., personal, home, auto).
    - ProductCategory: Category of loan.
    - Balance: Outstanding loan balance.
    - Term: Loan duration (e.g., 36 months).

6. Beneficiary

    - BeneficiaryID (PK): Unique identifier of the beneficiary.
    - CustomerID (FK → Customer.CustomerID): Customer who added the beneficiary.
    - AccountNumber: Beneficiary’s account number.
    - BankName: Beneficiary’s bank name.

**Relationships between tables**

1. Customer – Account
    * One-to-Many: A customer can hold multiple accounts.
    * FK: Account.CustomerID → Customer.CustomerID

2. Customer – Deposit
    * One-to-Many: A customer can hold multiple deposit products.
    * FK: Deposit.CustomerID → Customer.CustomerID

3. Customer – Loan
    * One-to-Many: A customer can take multiple loans.
    * FK: Loan.CustomerID → Customer.CustomerID

4. Customer – Beneficiary
    * One-to-Many: A customer can add multiple beneficiaries.
    * FK: Beneficiary.CustomerID → Customer.CustomerID

5. Account – Transactions
    * One-to-Many: An account can have multiple transactions.
    * FK: Transactions.AccountID → Account.AccountID

![](https://raw.githubusercontent.com/datascienceworld-kan/vinagent/refs/heads/main/docs/docs/get_started/images/financial_db.png)

**Figure 1.** The database schema of the banking database.

We will run SQL code to create a fake finanice database. The SQL code for the fake database is available at [banking.sql](https://github.com/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/guides/banking.sql). Let's download them and save to your local machine at `./banking.sql`.


```python
import os
import sqlite3

def create_banking_database(sql_table_name: str="", sql_script_path: str=""):
    if not os.path.exists(f"{sql_table_name}"):
        # Connect to SQLite database (creates financial_db.db if it doesn't exist)
        conn = sqlite3.connect(sql_table_name)
        cursor = conn.cursor()

        # Read the SQL file
        with open(sql_script_path, 'r') as file:
            sql_script = file.read()

        # Split the script into individual statements
        statements = sql_script.split(';')

        # Execute each SQL statement
        for statement in statements:
            # Skip empty statements
            if statement.strip():
                try:
                    cursor.execute(statement)
                except sqlite3.Error as e:
                    print(f"Error executing statement: {e}\nStatement: {statement}")

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        print(f"Database {sql_table_name} created and populated successfully.")
    else:
        print(f"Database {sql_table_name} already exists.")


sql_table_name = 'financial_db.db'
sql_script_path = './banking.sql' # You should select right path of SQL script in your local machine.
create_banking_database(sql_table_name, sql_script_path)
```

    Database financial_db.db created and populated successfully.


Let's test the connection to the new database.


```python
from vinagent.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{sql_table_name}")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Customer LIMIT 5;")}')
```

## Building SQL Database Toolkits

The heart of our agent lies in its SQL capabilities. Vinagent provides four essential SQL tools that work together:

- `sql_db_query`: Execute an input SQL query and returns a result from the database.

- `sql_db_schema`: Searching the table schemas and sample rows for the list of input tables. This tool helps LLM understand the context of list tables.

- `sql_db_list_tables`: List out the list of available tables in the database.

- `sql_db_query_checker`: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with `sql_db_query` to avoid database error.


```python
from vinagent.utilities import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

# Extract individual tools for our nodes
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
sql_db_list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
query_checker_tool = next(tool for tool in tools if tool.name == "sql_db_query_checker")

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
```

    sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
    
    sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3
    
    sql_db_list_tables: Input is an empty string, output is a comma-separated list of tables in the database.
    
    sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
    
## Creating Intelligent Agent Nodes

Now we'll build the individual components (nodes) that make up our agent's workflow. Each node has a specific responsibility:

### Node 1: Table discovery

This node identifies all available tables in the database:

```python
from typing import Annotated, TypedDict
from vinagent.graph.operator import FlowStateGraph, END, START
from vinagent.graph.node import Node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.utils.runnable import coerce_to_runnable
from langgraph.graph import MessagesState
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

class ListTablesNode(Node):
    def exec(self, state: MessagesState) -> dict:
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        
        list_tables_tool = sql_db_list_tables_tool
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [response]}
```

### Node 2: Schema Analysis

Not all tables will be selected to proceed with the user query, therefore, the next node will filter which tables are neccessary.


```python
class CallGetSchemaNode(Node):    
    def exec(self, state: MessagesState) -> dict:
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
```

Let's extract a list of such table schemas with relevant records as an additional context for SQL query generation.

```python
class GetSchemaNode(Node):
    def exec(self, state: MessagesState) -> dict:
        response = get_schema_tool.invoke(state['messages'][-1].tool_calls[0])
        return {"messages": [response]}
```

### Node 3: Query Generation and Execution

This is the brain of our agent because it generates SQL queries and decides when to execute them. Based on user query and the extracted table schema context, this node generates the SQL query to be executed by the next node. There are two scenarios to trigger next nodes:

1. If the workflow pipeline has not yet obtained the final answer, let's generate an SQL query at the first round or correct the incorrect SQL query if it was other rounds. Passing SQL query to `check_query` node to verify the query validation.

2. If the system has achieved the final output, just need to answer in natural language given the SQL output. Afterwards, come to the `END` node to finish.


```python
generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)

generate_answer_system_prompt = "Let's answer in natural format based on the SQL result."
```


```python
class GenerateQueryNode(Node):
    def exec(self, state: MessagesState) -> dict:
        llm_with_tools = llm.bind_tools([run_query_tool])
        last_message = state["messages"][-1]
        if (last_message.name == 'sql_db_query'):
            system_message = {
                "role": "system",
                "content": generate_answer_system_prompt,
            }
            last_message = AIMessage(
                content=last_message.content
            )
            response = llm.invoke([system_message] + state["messages"] + [last_message])
            return {"messages": [response]}
        else:
            system_message = {
                "role": "system",
                "content": generate_query_system_prompt,
            }
            response = llm_with_tools.invoke([system_message] + state["messages"])
            return {"messages": [response]}

    def branching(self, state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if len(last_message.tool_calls) > 0:
            return "check_query"
        else:
            return END
```

### Node 5: Checking Query

The next node presents a checking query step to ensure the correctness of generated SQL query. Otherwise, the incorrect execution can detain and slow down SQL execution engine.


```python
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)

class CheckQueryNode(Node):
    def exec(self, state: MessagesState) -> dict:
        system_message = {
            "role": "system",
            "content": check_query_system_prompt,
        }

        # Generate an artificial user message to check
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}
```

### Node 6: Run SQL Query

Finally, we can run generated SQL query to obtain the result if this SQL query was confirmed valid before.

```python
class RunQueryNode(Node):
    def exec(self, state: MessagesState) -> dict:
        response = run_query_tool.invoke(state['messages'][-1].tool_calls[0])
        return {"messages": [response]}
```

## Orchestrating the Agent Workflow

In this step, we will generate an AI Agent that orchestrates all nodes into a functional workflow for the text-to-SQL task. Let's see how this Agent understands the database schemas and relationships in order to provide precise answers for queries related to the banking and finance domain. The main pipeline is initialized using [FlowStateGraph](https://datascienceworld-kan.github.io/vinagent/get_started/workflow_and_agent/#flowstategraph)


```python
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from typing import Iterator
from langgraph.graph import MessagesState

# This configuration states the user_id. Therefore, it will remember who is chatting with Agent.
class ConfigSchema(TypedDict):
    user_id: str

class Agent:
    def __init__(self):
        self.checkpoint = MemorySaver()
        self.graph = FlowStateGraph(state_schema=MessagesState, config_schema=ConfigSchema)
        self.list_tables_node = ListTablesNode()
        self.call_get_schema_node = CallGetSchemaNode()
        self.get_schema_node = GetSchemaNode()
        self.generate_query_node = GenerateQueryNode()
        self.check_query_node = CheckQueryNode()
        self.run_query_node = RunQueryNode()

        self.flow = [
            self.list_tables_node >> self.call_get_schema_node,
            self.call_get_schema_node >> self.get_schema_node,
            self.get_schema_node >> self.generate_query_node,
            self.generate_query_node >> {
                "__end__": END, # For branching with END and START node, you only set key is __end__ and __start__
                "check_query": self.check_query_node
            },
            self.check_query_node >> self.run_query_node,
            self.run_query_node >> self.generate_query_node  # Loop back for multiple interactions
        ]

        self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint, flow=self.flow)

    def invoke(self, input_state: dict, config: dict) -> dict:
        return self.compiled_graph.invoke(input_state, config)

    def stream(self, input_state: dict, config: dict, **kwargs) -> Iterator[dict]:
        stream_mode = kwargs.get("stream_mode", "values")
            
        for state in self.compiled_graph.stream(input_state, config, stream_mode=stream_mode):
            if "messages" in state and state["messages"]:
                yield state
            else:
                continue

# Initialize input_state with the input question and thread config
question = "Who has the highest total loan balance?"
input_state = {
    "messages": [{"role": "user", "content": question}]
}
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}

# # Initialize agent and involve
agent = Agent()
agent.compiled_graph
```

![](https://raw.githubusercontent.com/datascienceworld-kan/vinagent/refs/heads/main/docs/docs/get_started/images/bank_agent.png)

```
result = agent.invoke(input_state=input_state, config=config)
for message in result['messages']:
    message.pretty_print()
```


    ================================[1m Human Message [0m=================================

    What is the customer has the highest total balance?
    ==================================[1m Ai Message [0m==================================

    Available tables: Account, Beneficiary, Customer, Deposit, Loan, Transactions
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
    sql_db_schema (call_uhDBc4fqf9NFjlSeUC7pK4jp)
    Call ID: call_uhDBc4fqf9NFjlSeUC7pK4jp
    Args:
        table_names: Account, Customer
    =================================[1m Tool Message [0m=================================
    Name: sql_db_schema


    CREATE TABLE "Account" (
        "AccountID" INTEGER, 
        "CustomerID" INTEGER, 
        "ProductType" VARCHAR(50), 
        "ProductCategory" VARCHAR(50), 
        "Balance" DECIMAL(15, 2) NOT NULL, 
        PRIMARY KEY ("AccountID"), 
        FOREIGN KEY("CustomerID") REFERENCES "Customer" ("CustomerID")
    )

    /*
    3 rows from Account table:
    AccountID	CustomerID	ProductType	ProductCategory	Balance
    101	1	Savings Account	Deposit Account	12000.00
    102	2	Current Account	Deposit Account	11500.00
    103	3	Credit	Credit Account	9500.00
    */


    CREATE TABLE "Customer" (
        "CustomerID" INTEGER, 
        "Name" VARCHAR(100) NOT NULL, 
        "Address" VARCHAR(255), 
        "Contact" VARCHAR(50), 
        "Username" VARCHAR(50) NOT NULL, 
        "Password" VARCHAR(255) NOT NULL, 
        PRIMARY KEY ("CustomerID")
    )

    /*
    3 rows from Customer table:
    CustomerID	Name	Address	Contact	Username	Password
    1	John Doe	123 Main St, Springfield, IL 62701	555-0101	johndoe	hashed_password1
    2	Jane Smith	456 Oak Ave, Boulder, CO 80302	555-0102	janesmith	hashed_password2
    3	Alice Johnson	789 Pine Rd, Asheville, NC 28801	555-0103	alicej	hashed_password3
    */
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
    sql_db_query (call_uvBiRWOGNQCrpmCUfoe4mBWC)
    Call ID: call_uvBiRWOGNQCrpmCUfoe4mBWC
    Args:
        query: SELECT c.CustomerID, c.Name, SUM(a.Balance) AS TotalBalance
    FROM Account a
    JOIN Customer c ON a.CustomerID = c.CustomerID
    GROUP BY c.CustomerID, c.Name
    ORDER BY TotalBalance DESC
    LIMIT 1;
    =================================[1m Tool Message [0m=================================
    Name: sql_db_query

    [(55, 'Caleb Price', 25000)]
    ==================================[1m Ai Message [0m==================================

    The customer with the highest total balance is:  
    • Customer ID: 55  
    • Name: Caleb Price  
    • Total Balance: 25,000


## Vinagent Agent with Workflow

Vinagent agent allows to integrate working flow as an attribute. To initialize agent, you just need to state three attributes:

- flow: a list of routes demonstates the working flow. Each route presents a strateforward route `start_node >> end_node` or a conditional route `start_node >> {'a': node_a, 'b': node_b}`, which will define the next node given the return result of `start_node`.

- state_schema: Define a state storage for agent workflow. This will save all intermediate messages which is returned at each node to access them after the graph execution is finished. By default, the state_schema is `langgraph.graph.MessageState`.

- config_schema: Define a config schema which pass before the workflow triggering to manage the `thread_id` and `user_id` the agent is chatting with.


```python
from langgraph.graph import MessagesState
from vinagent.agent import Agent

class ConfigSchema(TypedDict):
    user_id: str

list_tables_node = ListTablesNode()
call_get_schema_node = CallGetSchemaNode()
get_schema_node = GetSchemaNode()
generate_query_node = GenerateQueryNode()
check_query_node = CheckQueryNode()
run_query_node = RunQueryNode()
from langgraph.graph import MessagesState
from vinagent.agent import Agent

class ConfigSchema(TypedDict):
    user_id: str


def initialize_bank_agent():
    list_tables_node = ListTablesNode()
    call_get_schema_node = CallGetSchemaNode()
    get_schema_node = GetSchemaNode()
    generate_query_node = GenerateQueryNode()
    check_query_node = CheckQueryNode()
    run_query_node = RunQueryNode()

    agent = Agent(
        llm = llm,
        checkpoint = MemorySaver(),
        flow = [
            list_tables_node >> call_get_schema_node,
            call_get_schema_node >> get_schema_node,
            get_schema_node >> generate_query_node,
            generate_query_node >> {
                "__end__": END, # For branching with END and START node, you only set key is __end__ and __start__
                "check_query": check_query_node
            },
            check_query_node >> run_query_node,
            run_query_node >> generate_query_node  # Loop back for multiple interactions
        ],
        state_schema = MessagesState,
        config_schema = ConfigSchema,
    )
    return agent

bank_agent = initialize_bank_agent()
```


```python
question = "Who has the highest total loan balance?"
result = bank_agent.invoke(query=question)

for message in result['messages']:
    message.pretty_print()
```


# Advanced Features and Benefits

## Memory and Context Management
Our agent maintains conversation history, enabling follow-up questions:

```python
# Follow-up question (agent remembers context that include customer id) 
agent.invoke(query="How many transaction this cusomter have?")
```

## Asynchronous Processing
With the long-running and complex task like SQL pipeline, we should use asynchronous invoking to save the execution time as the following `ainvoke`.

```python
question = "Who has the highest total loan balance?"
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}

result = await bank_agent.ainvoke(query=question, config=config)
for message in result['messages']:
    print(message)
```

!!! note
    This `ainvoke` example is only suitable to run on Jupyter Notebook, where asynchronous execution is available. However, if you run on python module. You should cover your asynchronous method inside a `asyncio.run()` method.


```python
import asyncio

question = "Who has the highest total loan balance?"
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}

async def main():
    result = await bank_agent.ainvoke(query=question, config=config)
    return result

result = asyncio.run(main())
result
```

## Streaming for Real-time Updates
Or you can run under streaming mode, which facilitate to track and debug the intermedidate messages.

```python
question = "Who has the highest total loan balance?"
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}

for state in bank_agent.stream(query=question, config=config):
    if 'messages' in state:
        print(state['messages'])
    else:
        print(state)
```

# Test usercases

This section, we will test the banking agent across different use cases, starting with simple queries on a single table and progressing to more complex queries involving multiple tables.

## Usercase 1 - Find transaction history


```python
question = "Show the last 5 transactions of customer John Doe?"

bank_agent = initialize_bank_agent()
result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```


    Here are the most recent transactions for John Doe (Account #101). Since only two are recorded, these are the latest:
    
    TransactionID | AccountID | Type       | Amount   | Timestamp  
    ------------- | --------- | ---------- | -------- | -------------------  
    2             | 101       | Withdrawal | $500.00  | 2025-08-02 14:30:00  
    1             | 101       | Deposit    | $1,000.00| 2025-08-01 10:00:00


## Usercase 2 - Find customer have highest deposit balance


```python
question = "Which customer has the highest total deposit balance?"

bank_agent = initialize_bank_agent()
result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```


    The customer with the highest total deposit balance is David Lee (CustomerID 10) with a total of 19,700.00.


## Usercase 3 - Total loans, deposits, and accounts


```python
question = "What is the total balance and number of loans, deposits, and accounts in the bank?"

bank_agent = initialize_bank_agent()
result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```


    Here’s the summary by product category:
    
    • Accounts  
      – Number of accounts: 62  
      – Total balance: 499,100
    
    • Deposits  
      – Number of deposits: 128  
      – Total balance: 600,700
    
    • Loans  
      – Number of loans: 72  
      – Total balance: –946,500


## Usercase 4 - Find the total balance by products


```python
question = "What are the total balance of loans by product types?"

bank_agent = initialize_bank_agent()
result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```


    Here are the total loan balances, by product type:
    
    • Business Loan: –223 000  
    • Unsecured Loan: –239 500  
    • Secured Loan: –484 000



```python
question = "What are the total balance of loans by product categories?"

result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```

    Here are the total outstanding loan balances, grouped by product category:
    
    • Home Repair: –84,500.00  
    • Studying Loan: –155,000.00  
    • Car Loan: –157,000.00  
    • Household Business: –223,000.00  
    • Home Loan: –327,000.00  
    
    (All balances are negative, reflecting amounts owed.)


## Usercase 5 - Accounts, deposits, and loans by customer - Join multiple tables


```python
question = "What are total balance of accounts, loans, and deposits for each customer. Let's return a full table not limit rows?"

bank_agent = initialize_bank_agent()
result = bank_agent.invoke(query=question)
print(result['messages'][-1].content)
```

You can verify the original SQL query, which runs the output.


```python
print(result['messages'][-3].tool_calls[0]['args']['query'])
```

    SELECT c.CustomerID, c.Name, 
           COALESCE(acc.total_balance,0) AS total_account_balance,
           COALESCE(ln.total_balance,0) AS total_loan_balance,
           COALESCE(dep.total_balance,0) AS total_deposit_balance
    FROM Customer AS c
    LEFT JOIN (
      SELECT CustomerID, SUM(Balance) AS total_balance
      FROM Account
      GROUP BY CustomerID
    ) AS acc ON c.CustomerID = acc.CustomerID
    LEFT JOIN (
      SELECT CustomerID, SUM(Balance) AS total_balance
      FROM Loan
      GROUP BY CustomerID
    ) AS ln ON c.CustomerID = ln.CustomerID
    LEFT JOIN (
      SELECT CustomerID, SUM(Balance) AS total_balance
      FROM Deposit
      GROUP BY CustomerID
    ) AS dep ON c.CustomerID = dep.CustomerID
    ORDER BY c.CustomerID;

# Production Considerations
## Security Best Practices

- SQL Injection Prevention: You should have an [Authentication Layer](https://datascienceworld-kan.github.io/vinagent/get_started/authen_layer/) to prevent harmful SQL queries from invalid users.
- Access Control: Implement user-based table access restrictions.
- Audit Logging: Track all queries and results for compliance.
- Data Masking: Sensitive fields should be masked in responses.

## Performance Optimization

- Database Indexing: Ensure proper indexes on frequently queried columns
- Query Caching: Cache common query results to reduce database load
- Connection Pooling: Use connection pools for high-throughput scenarios
- Result Pagination: Implement pagination for large result sets

# Conclusion: Transforming Banking Operations
We've built a sophisticated banking agent that transforms natural language questions into actionable database insights. This agent offers several key advantages:

**For Business Users:**

- No SQL knowledge required
- Instant answers to complex questions
- Natural language interaction
- Real-time data access

**For IT Teams:**

- Reduced manual query writing
- Built-in security validations
- Scalable architecture
- Comprehensive audit trails

**For Organizations:**

- Faster decision-making
- Improved operational efficiency
- Better risk management
- Enhanced customer service

The agent we've built demonstrates how AI can bridge the gap between business needs and technical implementation, making data-driven decision-making accessible to everyone in a financial organization.
Whether you're analyzing customer portfolios, monitoring transaction patterns, or assessing risk exposure, this Vinagent-powered solution provides the intelligence and speed modern banking demands.
