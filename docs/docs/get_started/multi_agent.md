# Multi-Agent
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/multi_agent.ipynb)

May be, you are familiar with single agent design of Vinagent, which uses an LLM as brain to control tools and workflow of an application. However, as you develop many operational systems in many real enterprises, the operational procedure might be very complex, which requires a transformation from single-agent into multi-agent system over time. For example, you may have to tackle with the following challanges:

- Single agent increases it's workload, therefore, it need too many tools in use, which in return making a poor decision about which tool to call next.
- Context growing huge preventing agent from capturing the main important keypoints.
- Multiple specialization areas requires a complex system of multi-agent to deal with (e.g. planner, researcher, math expert, etc.)

Therefore, to tackle these, it is necessary to break your application into multiple agents, each agent has their own skills to be integrated into a multi-agent system. Comparing with single-agent, multi-agent system has a primary benefits are:

- Modularity: Separate agents simplify development and debugging.
- Specialization: Expert agents improve domain performance.
- Control: Explicit communication improves transparency and governance.

![](images/multi_agent_architectures.png)

Source: [Langchain blog](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

There are several ways to develop a multi-agent system:

- Network: All agents can talk directly with each other, and each one decides whom to call next.
- Supervisor: Agents only talk to a central supervisor, which decides the next agent to invoke.
- Supervisor (tool-calling): A variant where agents act like tools, and the supervisor LLM decides which tool-agent to use and what arguments to provide.
- Hierarchical: Supervisors can themselves have supervisors, enabling layered control structures.
- Custom workflow: Agents only connect with specific subsets, with some flows being fixed and others allowing limited decision-making.

In this tutorial, let's study how to develop a multi-agent system using Vinagent library.

## Setup

Multi-agent feature is supported from vinagent version `0.0.6`. Therefore, you should install upgradation version first:

```
!pip install vinagent=0.0.6
```

Initialize LLM model.


```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))

llm = ChatOpenAI(
    model = "o4-mini"
)
```

You should define `OPENAI_API_KEY` inside `.env` file.

## Multi-Agent in Vinangent

Vinagent designs an advanced multi-agent solution with key strengths:
- Specialized Agents: Each [single agent](https://datascienceworld-kan.github.io/vinagent/#component-overview) is fully equipped with its own LLM, tools, memory, skills, and authentication layer.
- Shared Conversation: Agents collaborate seamlessly in the same conversation, enabling them to capture and utilize each other’s context.
- Human-in-the-Loop: Users can directly participate and interact within the agent workflow.
- Customizable Order: A Crew class allows flexible control over the sequence of agents in a conversation.

## AgentNode

Each agent member in a multi-agent system is setup from [Vinagent's Agent class](https://datascienceworld-kan.github.io/vinagent/get_started/basic_agent/#setup-an-agent). However, to empower these agents to join in the same conversation, we specifically design a class `AgentNode` as a Proxy Class, which will be implemented for each agent. While initializing a new Agent class, you need to do following:

- Create a specific class inherites the AgentNode.
- Re-define `exec` method inside this class, which triggers invoking function (is one of `invoke, ainvoke, and stream`) inside. The behavior of triggers invoking function is similar as [Vinagent's Agent triggering](https://datascienceworld-kan.github.io/vinagent/get_started/async_invoke/)
- The conversation is recorded into a state, which is accessible for every members.

```python
class ExampleAgent(AgentNode):   
    def exec(self, state: State) -> dict:
        messages = state["messages"]
        output = self.invoke(messages)
        return {"messages": {"role": "AgentPositive", "content": output}}
```

To showcase the efficiency of Vinagent’s multi-agent system, we use the real-world case of customer service support on an e-commerce platform. The workflow follows this pipeline:

**Input → Supervisor Agent → [Negative | Positive | Neutral Agent] → User Feedback → Staff Agent → Output**

- Supervisor Agent: Analyzes customer comments on purchased products and classifies them as negative, positive, or neutral.

- Routing: Depending on the classification, the comment is forwarded to the corresponding agent (Negative, Positive, or Neutral).

- Negative Agent: Responds with an apology, collects user details (email, phone), and forwards them to the Staff Agent.

- Staff Agent: Applies customer care policies and sends a follow-up email to the customer.

- Positive & Neutral Agents: Since these cases are non-critical, they process the feedback accordingly without escalation.

![](images/customer_service_support.png)

Let's initialize each specific agent class by implementing `AgentNode`, each one should re-define `exec` method to deal with the return answer.


```python
from typing import Annotated, TypedDict
from vinagent.logger.logger import logging_message
from vinagent.multi_agent import AgentNode
from vinagent.multi_agent import CrewAgent

# Define a reducer for message history
def append_messages(existing: list, update: str) -> list:
    return existing + [update]

# Define the state schema
class State(TypedDict):
    messages: Annotated[list[str], append_messages]
    sentiment: str

# Define node classes
class Supervisor(AgentNode):
    @logging_message
    def exec(self, state: State) -> dict:
        message = state["messages"][-1]["content"]
        output = self.invoke(message)
        sentiment = 'neutral'
        if 'negative' in output.content.lower():
            sentiment = 'negative'
        elif 'positive' in output.content.lower():
            sentiment = 'positive'
        return {"messages": {"role": "Supervisor", "content": output}, "sentiment": sentiment}

    def branching(self, state: State) -> str:
        return state["sentiment"]

class AgentPositive(AgentNode):
    @logging_message
    def exec(self, state: State) -> dict:
        messages = state["messages"]
        output = self.invoke(messages)
        return {"messages": {"role": "AgentPositive", "content": output}}

class AgentNegative(AgentNode):
    @logging_message
    def exec(self, state: State) -> dict:
        messages = state["messages"]
        output = self.invoke(messages)
        return {"messages": {"role": "AgentNegative", "content": output}}

class AgentNeutral(AgentNode):
    @logging_message
    def exec(self, state: State) -> dict:
        messages = state["messages"]
        output = self.invoke(messages)
        return {"messages": {"role": "AgentNeutral", "content": output}}

class AgentStaff(AgentNode):
    @logging_message
    def exec(self, state: State) -> dict:
        messages = state["messages"]
        print(f'agent staff input messages: {messages}')
        output = self.invoke(messages)
        return {"messages": {"role": "AgentStaff", "content": output}}
```

Initializing the member agents join in the crew replying on their specific Agent class.


```python
supervisor = Supervisor(
    name="supervisor",
    description="A Supervisor agent who manage the task and assign it to your member agents",
    instruction="You only answer in one of three options: 'negative', 'positive', 'neutral'",
    llm=llm,
    skills=[
        "Classify user's query sentiment"
        "Assign task to member agents",
    ],
    tools=[
        'vinagent/tools/hello.py'
    ],
    memory_path="vinagent/templates/mutli_agent/supervisor/memory.json",
    tools_path = "vinagent/templates/mutli_agent/supervisor/tool.json"
)

agent_positive = AgentPositive(
    name="agent_positive",
    description="agent_positive agent process positive feedback",
    instruction="Customer is very happy let's thank you to them and ask them for rating",
    llm=llm,
    skills = [
        "Give thank's you to user",
    ],
    memory_path="vinagent/templates/mutli_agent/positive/memory.json",
    tools_path = "vinagent/templates/mutli_agent/positive/tool.json"
)

agent_negative = AgentNegative(
    name="agent_negative",
    description="agent_negative agent process negative feedback",
    instruction="Customer is unhappy with our service, let's show your sympathy with them, ask his information including email and number phone to forward to staff",
    llm=llm,
    skills = [
        "Give apology to user",
        "Asking for to make detailed complaints"
    ],
    memory_path="vinagent/templates/mutli_agent/negative/memory.json",
    tools_path = "vinagent/templates/mutli_agent/negative/tool.json"
)

agent_neutral = AgentNeutral(
    name="agent_neutral",
    description="agent_neutral agent process neutral feedback",
    instruction="You should respond by a decent utterance",
    llm=llm,
    skills = [
        "Understand customer intent and answer to them"
    ],
    memory_path="vinagent/templates/mutli_agent/neutral/memory.json",
    tools_path = "vinagent/templates/mutli_agent/neutral/tool.json"
)


agent_staff = AgentStaff(
    name="agent_staff",
    description="agent_staff to process customer complaints",
    instruction="Customer is unhappy with our service, let's analyze his complaints and write an sorry email to them with detailed compensation",
    llm=llm,
    skills = [
        "Give an apology email to user",
        "Confirm customer number phone again"
    ],
    tools=["vinagent/vinagent/tools/crm_system/apology_incorrect_delivery_email.py"],
    memory_path="vinagent/templates/mutli_agent/staff/memory.json",
    tools_path = "vinagent/templates/mutli_agent/staff/tool.json"
)
```

!!! note 
    Organizing each agent with its own dedicated folder for memory and tools creates a secure, isolated architecture that prevents interference and conflicts between agents. This separation ensures safety during updates since you can modify one agent's memory and tools without affecting others, while also enabling specialized customization for each agent's specific role. The folder-based approach optimizes performance by allowing agents to load only what they need and operate in parallel without stepping on each other's data. Most importantly, this structure provides robust security and safety guarantees, ensuring that each agent's sensitive data and specialized tools remain protected and accessible only when appropriate.

![](images/template_multi_agent.png)

_Hierarchiral structure seperates out each agent memory and tool accordingly._

## Human-in-the-loop

Humans can join the multi-agent system to provide feedback and messages. You should define the main information that the user inputs into the multi-agent system inside the `exec` method of the `UserFeedback` class. The following is a `user_feedback` instance that will be integrated into the crew agent.



```python
from vinagent.multi_agent import UserFeedback

class Feedback(UserFeedback):
    def exec(self, state: State) -> dict: # Must have state in argument
        email = input("Please provide your email:")
        phone = input("Please provide your phone:")
        output = f'Email: {email}; Phone: {phone}'
        return {"messages": {"role": "user", "content": output}}

user_feedback = Feedback(
    name="user_feedback",
    role="user"
)
```

## Crew of Agent

Crew class is group of agents who join in this conversatio



```python
from vinagent.graph.operator import FlowStateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Optional config schema
class ConfigSchema(TypedDict):
    user_id: str

# Initialize Crew
crew = CrewAgent(
    llm = llm,
    checkpoint = MemorySaver(),
    graph = FlowStateGraph(State, config_schema=ConfigSchema),
    flow = [
        supervisor >> {
            "positive": agent_positive,
            "neutral": agent_neutral,
            "negative": agent_negative,
        },
        agent_positive >> END,
        agent_neutral >> END,
        agent_negative >> user_feedback,
        user_feedback >> agent_staff,
        agent_staff >> END
    ]
)
```


```python
crew.compiled_graph
```
    
![png](images/multi_agent_15_0.png)
    
This generates a visual representation of your multi-agent workflow, helping you understand and debug the system architecture.

### Invoke

To kick-off multi-agent system, you only need to pass query inside `invoke()` method.


```python
query="I'm not happy about this product. I ordered 5, but you delivered 4 items. The paper wrapper is torn out."
result = crew.invoke(query=query, user_id="Kan", thread_id=123)
```

!!! note 
    In there, each user will have their own identification by `user_id`. Each conversation should be kicked-off
    inside a specific `thread_id` to support parallel executions while there many requests to crew in the production
    environment. The default value of `thread_id` is `123`.


```python
for mess in result['messages']:
    content=mess['content'].content if hasattr(mess['content'], "content") else mess['content']
    print(f"======== {mess['role']} Response ========\n{content}\n\n")
```

    ======== user Response ========
    I'm not happy about this product. I ordered 5, but you delivered 4 items. The paper wrapper is torn out.
    
    
    ======== Supervisor Response ========
    negative
    
    
    ======== AgentNegative Response ========
    Hello unknown_user, I’m very sorry to hear about this. It sounds like you ordered 5 items but only received 4, and the paper wrapper arrived torn. I understand how frustrating that must be. To get this resolved as quickly as possible, could you please provide:
    
    • Your order number  
    • Your email address  
    • A phone number where we can reach you  
    
    Once we have that information, I’ll forward everything to our support team right away. Thank you for your patience, and again, my apologies for the trouble.
    
    
    ======== user Response ========
    Email: vippro_customer@gmail.com; Phone: 849468686868
    
    
    ======== AgentStaff Response ========
    Subject: Apology and Resolution for Your Recent Order  
    To: vippro_customer@gmail.com  
    Phone: 849468686868  
    
    Dear Valued Customer,  
    
    I’m very sorry to hear that you received only four of the five items you ordered and that the paper wrapper arrived torn. That’s not the experience we strive to deliver, and I understand how frustrating this must be.  
    
    To make things right, here’s what we’d like to do:  
    1. Immediate Replacement  
       – We will ship the missing item at no additional cost to you.  
       – Your replacement will be sent via expedited shipping, on us, so you receive it as quickly as possible.  
    2. Refund Option  
       – If you’d rather receive a refund for the missing item instead of a replacement, please let us know and we’ll process it immediately.  
    3. Goodwill Discount  
       – As an apology for the inconvenience, we’d like to offer you a 15% discount on your next purchase. You can use code SORRY15 at checkout anytime over the next six months.  
    
    Once you let us know which option you prefer, we’ll have the replacement shipped or the refund issued within 24 hours and send you confirmation.  
    
    Again, I apologize for the trouble and appreciate your patience. Thank you for giving us the chance to make this right.  
    
    Sincerely,  
    Vippro Ecommerce Platform  
    support@vippro_ecm.com | 1-800-123-4567
    
    


### Asynchronously Invoke

For optimal performance with I/O-intensive operations:

```python
query="I'm not happy about this product. I ordered 5, but you delivered 4 items. The paper wrapper is torn out."
result = await crew.ainvoke(query=query, user_id="Kan", thread_id=123)
```

Asynchronous execution is particularly beneficial when handling external API calls, database operations, or multiple concurrent requests.


### Streaming

Monitor real-time agent interactions:


```python
for message in crew.stream(query=query, user_id="Kan", thread_id=123):
    print(message)
```
Streaming provides visibility into the multi-agent process, enabling real-time monitoring and debugging.

## Best Practices and Considerations

Indeed, there many real use cases that requires multi-agent architect, we can design using vinagent library. To have a good use of this feature, let's thoroughtly consider the following aspects before proceeding with your design:

- Design Guidelines

    - Clear Responsibilities: Define distinct roles for each agent to avoid overlap and confusion
    - State Management: Design your state schema to capture all necessary information for agent coordination
    - Error Handling: Implement robust error handling within each agent's exec method
    - Testing Strategy: Test individual agents before integrating them into the crew system

- Performance Optimization

    - Asynchronous Operations: Use ainvoke for I/O-bound operations
    - Memory Management: Configure appropriate memory paths for each agent
    - Tool Organization: Organize tools logically and avoid redundancy across agents

- Production Readiness

    - User Identification: Implement proper user_id management for multi-tenant applications
    - Thread Management: Use unique thread_ids for concurrent conversations
    - Monitoring: Leverage streaming capabilities for system monitoring and logging

## Conclusion

Vinagent's multi-agent system provides a powerful framework for building sophisticated, scalable applications. By breaking complex workflows into specialized agents, you can create more maintainable, efficient, and transparent systems.

The customer service example demonstrates how real-world business processes can be transformed into collaborative agent workflows, combining the strengths of specialized AI agents with human oversight and intervention capabilities.
Start with simple multi-agent configurations and gradually increase complexity as you become more familiar with the patterns and capabilities. The modular nature of Vinagent's approach ensures that your multi-agent systems can evolve and scale with your application's needs.
