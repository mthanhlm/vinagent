from typing_extensions import override
from vinagent.agent.agent import Agent
import json


class ReactAgent(Agent):
    @override
    def prompt_template(
        self, query: str, user_id: str = "unknown_user", *args, **kwargs
    ) -> str:
        try:
            tools = json.loads(self.tools_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        memory = ""
        if self.memory:
            memory_content = self.memory.load_memory_by_user(
                load_type="string", user_id=user_id
            )
            if memory_content:
                memory = f"- Memory: {memory_content}\n"
        else:
            memory = "- Memory: None\n"
        tools = self.tools_manager.load_tools()
        tool_names = list(tools.keys())
        prompt = (
            "Answer the following questions as best you can. You have access to the following tools:\n"
            f"{tools}\n"
            "Let's consider to memory and user_id:\n"
            f"{memory}\n"
            f"- user_id: {user_id}\n"
            "Use the following format:\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            f"Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "You must define input according to the following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"tool_type": "Type of tool. Only get one of three values ["function", "module", "mcp"]"\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n"
            "Begin!\n"
            f"Question: {query}\n"
            "Thought:"
        )
        return prompt
