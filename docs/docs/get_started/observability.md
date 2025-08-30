# Agent Observability

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/observability.ipynb)

Vinagent integrates with a local MLflow dashboard that can be used to visualize the intermediate messsages of each query. Therefore, it is an important feature for debugging.

- Engineer can trace the number of tokens, execution time, type of tool, and status of exection.
- Based on tracked results, Agent developers can indentify inefficient steps. Afterwards, optimize agent components like tools, prompts, agent description, agent skills, and LLM model.
- Accelerate process of debugging and improving the agent's performance.

Local tracing and observability ensure system security and data privacy, as your agent states are not dispatched outside your on-premise system. A local server can be quickly set up without creating an account, helping to reduce costs and accelerate the profiling process. Furthermore, Vinagent allows users to intervene in the logging states by adjusting the `vinagent.mlflow.autolog` code, enabling the addition of more state fields as needed.

Let's install vinagent library for this tutorial.

```python
%pip install vinagent
```

## Start MLflow UI

MLflow offers an local UI, which connets to mlflow server understreaming. This UI comprises all experients from conversations between user and agent. To start this UI, let's run this command on `terminal/command line interface` in your computer:


```python
mlflow ui
```

An MLflow dashboard starts, which can be accessed at [http://localhost:5000](http://localhost:5000).


## Initialize Experiment

Initialize an experiment to auto-log messages for agent


```python
import mlflow
from vinagent.mlflow import autolog

# Enable Vinagent autologging
autolog.autolog()

# Optional: Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("agent-dev")
```

    <Experiment: artifact_location='mlflow-artifacts:/451007843634367037', creation_time=1751455754824, experiment_id='451007843634367037', last_update_time=1751455754824, lifecycle_stage='active', name='agent-dev', tags={}>


After this step, an experiment named `agent-dev` is initialized. An observability and tracing feature are automatically registered for each query to the agent without requiring any changes to the original invocation code.

## Observability and Tracing

A default MLflow dashboard is launched to display the experiment results, within the Jupyter Notebook, making it convenient for agent developers to test and optimize their agent design directly. Every query is now tracked under the experiment named `agent-dev`.


```python
%%writefile .env
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are an Expert who can answer any general questions.",
    llm = llm,
    skills = [
        "Searching information from external search engine\n",
        "Summarize the main information\n"],
    tools = ['vinagent.tools.websearch_tools'],
    tools_path = 'templates/tools.json',
    memory_path = 'templates/memory.json'
)

result = agent.invoke(query="What is the weather today in Ha Noi?")
```

!!! note
    You are able to access the dashboard at http://localhost:5000/ and view logs of aformentioned query by accessing to `agent-dev` and click to `Traces` tab on the last of header navigation bar of `agent-dev` experiment.


<div>
  <style scoped>
  button {
    border: none;
    border-radius: 4px;
    background-color: rgb(34, 114, 180);
    font-family: -apple-system, "system-ui", "Segoe UI", Roboto, "Helvetica Neue", Arial;
    font-size: 13px;
    color: white;
    margin-top: 8px;
    margin-bottom: 8px;
    padding: 8px 16px;
    cursor: pointer;
  }
  button:hover {
    background-color: rgb(66, 153, 224);
  }
  </style>
  <button
    onclick="
        const display = this.nextElementSibling.style.display;
        const isCollapsed = display === 'none';
        this.nextElementSibling.style.display = isCollapsed ? null : 'none';

        const verb = isCollapsed ? 'Collapse' : 'Expand';
        this.innerText = `${verb} MLflow Trace`;
    "
  >Collapse MLflow Trace</button>
  <iframe
    id="trace-renderer"
    style="width: 100%; height: 500px; border: none; resize: vertical;"
    src="http://localhost:5000/static-files/lib/notebook-trace-renderer/index.html?trace_id=74105e3849a041e09eb828d73dfbf515&amp;experiment_id=900548938729976750&amp;trace_id=6ec5f26c820340ad9a2b5d3091771eb6&amp;experiment_id=900548938729976750&amp;version=3.1.0"
  />
</div>
