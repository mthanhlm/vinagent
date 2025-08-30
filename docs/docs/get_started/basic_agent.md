# Build a basic Chatbot
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/basic_agent.ipynb)

This tutorial introduce you how to create a simple Agent with minimal components and how to use them. This offers a general view on agent initialization and tool integration.

## Installation
The python distribution version of Vinagent library is avaible on pypi.org channel and github, which facilitates the installation of the library.

**Dev version on git**

You can clone git repository and install by poetry command. This is suitable to obtain the latest development version.

```
git@github.com:datascienceworld-kan/vinagent.git
cd vinagent
pip install -r requirements.txt
poetry install
```

**Stable version**

You can install the stable distributed versions which are tested and distributed on pypi.org channel by pip command

```
pip install vinagent
```

## Prerequisites
To use a list of default tools inside [vinagent.tools](https://github.com/datascienceworld-kan/vinagent/tree/main/vinagent/tools) you should set environment varibles inside `.env` including `TOGETHER_API_KEY` to use llm models at [togetherai](https://api.together.ai/signin) site and `TAVILY_API_KEY` to use tavily websearch tool at [tavily](https://app.tavily.com/home) site:

```
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```
Let's create your acounts first and then create your relevant key for each website.

## Setup an Agent
`vinagent` is a flexible library for creating intelligent agents. You can configure your agent with tools, each encapsulated in a Python module under `vinagent.tools`. This provides a workspace of tools that agents can use to interact with and operate in the realistic world. Each tool is a Python file with full documentation and it can be independently ran. For example, the [vinagent.tools.websearch_tools](vinagent/tools/websearch_tools.py) module contains code for interacting with a search API.


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 1: Create Agent with tools
agent = Agent(
    description="You are a Financial Analyst",
    llm = llm,
    skills = [
        "Deeply analyzing financial markets", 
        "Searching information about stock price",
        "Visualization about stock price"]
)
```

In there:

- description: The general description of Agent you want to build. For instance, who you are and how do you work?

- llm: The language model you want to use. You can use any models which are initialized by [OpenAI](https://github.com/openai/openai-python) class.

- skills: A list of skills you want your agent to have. Each skill is a string. Each agent will has particular skills. For example, if you want your agent to be a Financial Analyst, you can set your skills as `["Deeply analyzing financial markets", "Searching information about stock price"]`.

```
# Step 2: invoke the agent
message = agent.invoke("Who you are?")
print(message)
```

If the answer is a normal message without using any tools, it will be an `AIMessage`. By contrast, it will have `ToolMessage` type. For examples:

```
AIMessage(content='I am a Financial Analyst.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 308, 'total_tokens': 315, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-070f7431-7176-42a8-ab47-ed83657c9463-0', usage_metadata={'input_tokens': 308, 'output_tokens': 7, 'total_tokens': 315, 'input_token_details': {}, 'output_token_details': {}})
```
Access to `content` property to get the string content.

```
message.content
```
```
I am a Financial Analyst.
```
