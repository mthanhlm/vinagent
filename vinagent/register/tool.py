import sys
import os
import json
import inspect
import importlib
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union, Literal
import ast
import uuid
from pathlib import Path
import shutil
from vinagent.mcp import load_mcp_tools
from vinagent.mcp.client import DistributedMCPClient
from langchain_core.messages.tool import ToolMessage
from langchain_core.language_models.base import BaseLanguageModel
import asyncio
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolManager:
    """
    Centralized tool management class for registering, loading, saving, and executing tools.
    Tools are stored in a JSON file and can be of type 'function', 'mcp', or 'module'.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools: bool = False,
    ):
        """
        Initialize the ToolManager with a path to the tools JSON file.

        Args:
            llm (BaseLanguageModel): Language model instance for tool analysis.
            tools_path (Path, optional): Path to the JSON file for storing tools. Defaults to Path("templates/tools.json").
            is_reset_tools (bool, optional): If True, resets the tools file to an empty JSON object. Defaults to False.

        Behavior:
            - Converts tools_path to a Path object if provided as a string.
            - Creates the tools file if it does not exist.
            - Resets the tools file if is_reset_tools is True.
        """
        self.llm = llm
        self.tools_path = tools_path
        self.is_reset_tools = is_reset_tools
        self.tools_path = (
            Path(tools_path) if isinstance(tools_path, str) else tools_path
        )
        if not self.tools_path.exists():
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        if self.is_reset_tools:
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        self._registered_functions: Dict[str, Callable] = {}

    def load_tools(self) -> Dict[str, Any]:
        """
        Load existing tools from the JSON file.

        Returns:
            Dict[str, Any]: A dictionary of tool metadata, where keys are tool names.
        """
        if self.tools_path:
            with open(self.tools_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def save_tools(self, tools: Dict[str, Any]) -> None:
        """
        Save tools metadata to the JSON file.

        Args:
            tools (Dict[str, Any]): Dictionary of tool metadata to save.
        """
        with open(self.tools_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)

    def register_function_tool(self, func):
        """
        Decorator to register a function as a tool.

        Args:
            func: The function to register as a tool.

        Returns:
            Callable: A wrapped function that retains original behavior.

        Example:
            @tool_manager.register_function_tool
            def sample_function(x: int, y: str) -> str:
                '''Sample function for testing'''
                return f"{y}: {x}"

        Behavior:
            - Extracts function metadata (name, arguments, return type, docstring).
            - Assigns a unique tool_call_id.
            - Stores metadata in the tools JSON file.
            - Registers the function in _registered_functions for execution.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Get function metadata
        signature = inspect.signature(func)

        # Try to get module path, fall back to None if not available
        module_path = "__runtime__"

        # Create metadata
        if module_path == "__runtime__":
            metadata = {
                "tool_name": func.__name__,
                "arguments": {
                    name: (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    )
                    for name, param in signature.parameters.items()
                },
                "return": (
                    str(signature.return_annotation)
                    if signature.return_annotation != inspect.Signature.empty
                    else "Any"
                ),
                "docstring": (func.__doc__ or "").strip(),
                "module_path": module_path,
                "tool_type": "function",
                "tool_call_id": "tool_" + str(uuid.uuid4()),
                "is_runtime": module_path == "__runtime__",
            }

            # Register both the function and its metadata
            self._registered_functions[func.__name__] = func
            tools = self.load_tools()
            tools[func.__name__] = metadata
            self.save_tools(tools)
            logger.info(
                f"Registered tool: {func.__name__} "
                f"({'runtime' if module_path == '__runtime__' else 'file-based'})"
            )
        return wrapper

    async def register_mcp_tool(
        self, client: DistributedMCPClient, server_name: str = None
    ) -> list[Dict[str, Any]]:
        """
        Register tools from an MCP (Memory Compute Platform) server.

        Args:
            client (DistributedMCPClient): Client for interacting with the MCP server.
            server_name (str, optional): Name of the MCP server. Defaults to None.

        Returns:
            list[Dict[str, Any]]: List of registered MCP tool metadata.

        Behavior:
            - Fetches tools from the MCP server using the client.
            - Converts MCP tools to the internal tool format.
            - Assigns unique tool_call_id for each tool.
            - Saves tools to the JSON file.
        """
        logger.info(f"Registering MCP tools")
        all_tools = []
        if server_name:
            all_tools = await client.get_tools(server_name=server_name)
            logger.info(f"Loaded MCP tools of {server_name}: {len(all_tools)}")
        else:
            try:
                all_tools = await client.get_tools()
                logger.info(f"Loaded MCP tools: {len(all_tools)}")
            except Exception as e:
                logger.error(f"Error loading MCP tools: {e}")
                return []

        # Convert MCP tools to our format
        def convert_mcp_tool(mcp_tool: Dict[str, Any]):
            tool_name = mcp_tool["name"]
            arguments = dict(
                [
                    (k, v["type"])
                    for (k, v) in mcp_tool["args_schema"]["properties"].items()
                ]
            )
            docstring = mcp_tool["description"]
            return_value = mcp_tool["response_format"]
            tool = {}
            tool["tool_name"] = tool_name
            tool["arguments"] = arguments
            tool["return"] = return_value
            tool["docstring"] = docstring
            tool["module_path"] = "__mcp__"
            tool["tool_type"] = "mcp"
            # tool['mcp_client_connections'] = client.connections
            # tool['mcp_server_name'] = server_name
            tool["tool_call_id"] = "tool_" + str(uuid.uuid4())
            return tool

        new_tools = [convert_mcp_tool(mcp_tool.__dict__) for mcp_tool in all_tools]
        tools = self.load_tools()
        for tool in new_tools:
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")
        self.save_tools(tools)
        logger.info(f"Completed registration for mcp module {server_name}")
        return new_tools

    def register_module_tool(self, module_path: str) -> None:
        """
        Register tools from a Python module.

        Args:
            module_path (str): Path to the module or import path in module import format.

        Raises:
            ValueError: If the module cannot be loaded or tool format is invalid.

        Behavior:
            - Copies the module file to the tools directory if a file path is provided.
            - Imports the module and extracts tool metadata using the language model.
            - Assigns a unique tool_call_id for each tool.
            - Saves tools to the JSON file.
        """
        try:
            if os.path.isfile(module_path):
                # This is a path of module import format
                module_path = Path(module_path)
                absolute_lib_path = Path(os.path.dirname(os.path.abspath(__file__)))
                destination_path = Path(
                    os.path.join(absolute_lib_path.parent, "tools", module_path.name)
                )
                if module_path.resolve(strict=False) == destination_path.resolve(
                    strict=False
                ):
                    pass
                else:
                    shutil.copy2(module_path, destination_path)
                module_path = f"vinagent.tools.{destination_path.name.split('.')[0]}"
            module = importlib.import_module(module_path, package=__package__)
            module_source = inspect.getsource(module)
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load module {module_path}: {str(e)}")

        prompt = (
            "Analyze this module and return a list of tools in JSON format:\n"
            "- Module code:\n"
            f"{module_source}\n"
            "- Extract only tools marked with the @primary_function decorator. For example @primary_function def function_name(): ...\n"
            "- Let's return a list of json format without further explaination and without ```json characters markdown and keep module_path unchange.\n"
            "- Return value must be able to convert into a list from string.\n"
            "[{{\n"
            '"tool_name": "The function",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool. Let\'s keep default value if it was set",\n'
            '"return": "Return value of this tool",\n'
            '"docstring": "Docstring of this tool",\n'
            '"dependencies": "List of libraries need to run this tool",\n'
            f'"module_path": "{module_path}"\n'
            "}}]\n"
        )

        response = self.llm.invoke(prompt)

        response_text = response.content.strip()

        # Remove markdown code fences if present
        if response_text.startswith("```"):
            # Remove the first line (```json or ```)
            response_lines = response_text.splitlines()
            # Skip first line if it starts with ```
            if response_lines[0].startswith("```"):
                response_lines = response_lines[1:]
            # Remove last line if it's ```
            if response_lines and response_lines[-1].startswith("```"):
                response_lines = response_lines[:-1]
            response_text = "\n".join(response_lines)

        # Attempt to parse the entire text first
        try:
            new_tools = ast.literal_eval(response_text)
        except (ValueError, SyntaxError):
            # Fallback: extract the first JSON object/list from text
            extracted = self.extract_tool(response_text)
            print(f"extracted: {extracted}")
            if extracted:
                try:
                    new_tools = ast.literal_eval(extracted)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"Invalid tool format from LLM after extraction: {str(e)}"
                    )
            else:
                raise ValueError(
                    "Invalid tool format from LLM: could not find valid JSON list"
                )

        # Ensure new_tools is a list of dictionaries
        if isinstance(new_tools, dict):
            new_tools = [new_tools]
        if not isinstance(new_tools, list):
            raise ValueError(
                f"Invalid tool format from LLM: Expected list or dict, got {type(new_tools)}"
            )
        for idx, item in enumerate(new_tools):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid tool format from LLM: Element at index {idx} is {type(item)}, expected dict"
                )

        # Fallback to local introspection if required keys are missing or list is empty
        REQUIRED_KEYS = {"tool_name", "arguments", "return", "docstring"}

        def _introspect_module(module_obj, module_path_str):
            result = []
            for name, obj in inspect.getmembers(module_obj, inspect.isfunction):
                if inspect.getmodule(obj) != module_obj:
                    continue  # Skip imported functions
                sig = inspect.signature(obj)
                arguments = {
                    param_name: (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    )
                    for param_name, param in sig.parameters.items()
                }
                return_type = (
                    str(sig.return_annotation)
                    if sig.return_annotation != inspect.Signature.empty
                    else "Any"
                )
                metadata = {
                    "tool_name": name,
                    "arguments": arguments,
                    "return": return_type,
                    "docstring": (obj.__doc__ or "").strip(),
                    "dependencies": [],
                    "module_path": module_path_str,
                }
                result.append(metadata)
            return result

        if len(new_tools) == 0 or any(
            not REQUIRED_KEYS.issubset(item.keys()) for item in new_tools
        ):
            logger.warning(
                "LLM did not return valid tool metadata, falling back to introspection."
            )
            new_tools = _introspect_module(module, module_path)

        tools = self.load_tools()
        for tool in new_tools:
            tool["module_path"] = module_path
            tool["tool_type"] = "module"
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")

        self.save_tools(tools)
        logger.info(f"Completed registration for module {module_path}")

    def extract_tool(self, text: str) -> Optional[str]:
        """
        Extract the first valid JSON object from a text string.

        Args:
            text (str): The text to parse for a JSON object.

        Returns:
            Optional[str]: The extracted JSON string, or None if no valid JSON is found.
        """
        stack = []
        start = text.find("{")
        if start == -1:
            return None

        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
        module_path: str,
        tool_type: str = Literal["function", "mcp", "module"],
    ) -> Any:
        """
        Execute the specified tool with the given arguments.

        Args:
            tool_name (str): Name of the tool to execute.
            arguments (dict): Dictionary of arguments to pass to the tool.
            mcp_client (DistributedMCPClient): Client for MCP tool execution.
            mcp_server_name (str): Name of the MCP server.
            module_path (str): Path to the module for module-type tools.
            tool_type (str): Type of tool ('function', 'mcp', or 'module').

        Returns:
            Any: The result of the tool execution, typically a ToolMessage.
        """
        if tool_type == "function":
            message = await FunctionTool.execute(self, tool_name, arguments)
        elif tool_type == "mcp":
            message = await MCPTool.execute(
                self, tool_name, arguments, mcp_client, mcp_server_name
            )
        elif tool_type == "module":
            message = await ModuleTool.execute(self, tool_name, arguments, module_path)
        return message

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        Extract the first valid JSON object from text using stack-based parsing.

        Args:
            text (str): The text to parse for a JSON object.

        Returns:
            Optional[str]: The extracted JSON string, or None if no valid JSON is found.
        """
        start = text.find("{")
        if start == -1:
            return None

        stack = []
        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None


class FunctionTool:
    """
    Utility class for executing function-type tools.
    """

    @classmethod
    async def execute(
        cls, tool_manager: ToolManager, tool_name: str, arguments: Dict[str, Any]
    ):
        """
        Execute a registered function tool.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the function tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the function.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            Exception: If the function execution fails, logs the error and returns a message.
        """
        registered_functions = tool_manager.load_tools()

        if tool_name in tool_manager._registered_functions:
            try:
                func = tool_manager._registered_functions[tool_name]
                # artifact = await func(**arguments)
                artifact = await asyncio.to_thread(func, **arguments)
                content = f"Completed executing function tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions[tool_name]["tool_call_id"]
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = f"Failed to execute function tool {tool_name}({arguments}): {str(e)}"
                logger.error(content)
                # raise {"error": content}
                return content


class MCPTool:
    """
    Utility class for executing MCP-type tools.
    """

    @classmethod
    async def execute(
        cls,
        tool_manager: ToolManager,
        tool_name: str,
        arguments: Dict[str, Any],
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
    ):
        """
        Execute an MCP tool using the provided client and server.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the MCP tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            mcp_client (DistributedMCPClient): Client for interacting with the MCP server.
            mcp_server_name (str): Name of the MCP server.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            Exception: If the tool execution fails, logs the error and returns a message.
        """
        registered_functions = tool_manager.load_tools()
        """Call the MCP tool natively using the client session."""
        async with mcp_client.session(mcp_server_name) as session:
            payload = {"name": tool_name, "arguments": arguments}
            try:
                # Send the request to the MCP server
                # response = await session.call_tool(**payload)
                response = await session.call_tool(**payload)
                content = f"Completed executing mcp tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions[tool_name]["tool_call_id"]
                artifact = response
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = (
                    f"Failed to execute mcp tool {tool_name}({arguments}): {str(e)}"
                )
                logger.error(content)
                # raise {"error": content}
                return content


class ModuleTool:
    """
    Utility class for executing module-type tools.
    """

    @classmethod
    async def execute(
        cls,
        tool_manager: ToolManager,
        tool_name: str,
        arguments: Dict[str, Any],
        module_path: Union[str, Path],
        *arg,
        **kwargs,
    ):
        """
        Execute a module-based tool by importing and calling the specified function.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the module tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            module_path (Union[str, Path]): Path to the module containing the tool.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            ImportError, AttributeError: If the module or function cannot be loaded, logs the error and returns a message.
        """
        registered_functions = tool_manager.load_tools()
        try:
            if tool_name in globals():
                return globals()[tool_name](**arguments)

            module = importlib.import_module(module_path, package=__package__)
            func = getattr(module, tool_name)
            # artifact = await func(**arguments)
            artifact = await asyncio.to_thread(func, **arguments)
            content = f"Completed executing module tool {tool_name}({arguments})"
            logger.info(content)
            tool_call_id = registered_functions[tool_name]["tool_call_id"]
            message = ToolMessage(
                content=content, artifact=artifact, tool_call_id=tool_call_id
            )
            return message
        except (ImportError, AttributeError) as e:
            content = (
                f"Failed to execute module tool {tool_name}({arguments}): {str(e)}"
            )
            logger.error(content)
            # raise {"error": content}
            return content
