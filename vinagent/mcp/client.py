from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from .prompts import load_mcp_prompt
from .resources import load_mcp_resources
from .sessions import (
    Connection,
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
    create_session,
)
from .tools import load_mcp_tools

ASYNC_CONTEXT_MANAGER_ERROR = (
    "DistributedMCPClient cannot be used as a context manager (e.g., async with DistributedMCPClient(...)). "
    "Instead, you can do one of the following:\n"
    "1. client = DistributedMCPClient(...)\n"
    "   tools = await client.get_tools()\n"
    "2. client = DistributedMCPClient(...)\n"
    "   async with client.session(server_name) as session:\n"
    "       tools = await load_mcp_tools(session)"
)


class DistributedMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools, prompts and resources from them."""

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
    ) -> None:
        """Initialize a DistributedMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                If None, no initial connections are established.

        Example: basic usage (starting a new session on each tool call)

        ```python
        from vinagent.mcp.client import DistributedMCPClient

        client = DistributedMCPClient(
            {
                "math": {
                    "command": "python",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["/path/to/math_server.py"],
                    "transport": "stdio",
                },
                "weather": {
                    # make sure you start your weather server on port 8000
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        all_tools = await client.get_tools()
        ```

        Example: explicitly starting a session

        ```python
        from vinagent.mcp.client import DistributedMCPClient
        from vinagent.mcp.tools import load_mcp_tools

        client = DistributedMCPClient({...})
        async with client.session("math") as session:
            tools = await load_mcp_tools(session)
        ```
        """
        self.connections: dict[str, Connection] = (
            connections if connections is not None else {}
        )

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server and initialize a session.

        Args:
            server_name: Name to identify this server connection
            auto_initialize: Whether to automatically initialize the session

        Raises:
            ValueError: If the server name is not found in the connections

        Yields:
            An initialized ClientSession
        """
        if server_name not in self.connections:
            raise ValueError(
                f"Couldn't find a server with name '{server_name}', expected one of '{list(self.connections.keys())}'"
            )

        async with create_session(self.connections[server_name]) as session:
            if auto_initialize:
                await session.initialize()
            yield session

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Get a list of all tools from all connected servers.

        Args:
            server_name: Optional name of the server to get tools from.
                If None, all tools from all servers will be returned (default).

        NOTE: a new session will be created for each tool call

        Returns:
            A list of LangChain tools
        """
        if server_name is not None:
            if server_name not in self.connections:
                raise ValueError(
                    f"Couldn't find a server with name '{server_name}', expected one of '{list(self.connections.keys())}'"
                )
            return await load_mcp_tools(None, connection=self.connections[server_name])

        all_tools: list[BaseTool] = []
        for connection in self.connections.values():
            tools = await load_mcp_tools(None, connection=connection)
            all_tools.extend(tools)
        return all_tools

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> list[HumanMessage | AIMessage]:
        """Get a prompt from a given MCP server."""
        async with self.session(server_name) as session:
            prompt = await load_mcp_prompt(session, prompt_name, arguments=arguments)
            return prompt

    async def get_resources(
        self, server_name: str, *, uris: str | list[str] | None = None
    ) -> list[Blob]:
        """Get resources from a given MCP server.

        Args:
            server_name: Name of the server to get resources from
            uris: Optional resource URI or list of URIs to load. If not provided, all resources will be loaded.

        Returns:
            A list of LangChain Blobs
        """
        async with self.session(server_name) as session:
            resources = await load_mcp_resources(session, uris=uris)
            return resources

    async def __aenter__(self) -> "DistributedMCPClient":
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)

    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)


__all__ = [
    "DistributedMCPClient",
    "SSEConnection",
    "StdioConnection",
    "StreamableHttpConnection",
    "WebsocketConnection",
]
