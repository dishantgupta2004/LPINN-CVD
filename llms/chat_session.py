"""
Chat session management for LLM interactions.
"""
from typing import Dict, List, Optional, Union, Any
import json
import time
from dataclasses import dataclass, field

from llms.api_client import LLMClient
from llms.prompt_templates import CVD_SYSTEM_PROMPT
from llms.tools import CVDPINNTools, register_tools


@dataclass
class Message:
    """Message in a chat session."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[Dict[str, Any]] = None


class ChatSession:
    """
    Manages a chat session with an LLM, including history and tool usage.
    """
    
    def __init__(self, 
               llm_client: Optional[LLMClient] = None,
               tools: Optional[CVDPINNTools] = None,
               system_message: Optional[str] = None):
        """
        Initialize a chat session.
        
        Args:
            llm_client: LLM client instance
            tools: CVD-PINN tools instance
            system_message: Optional system message to override default
        """
        self.llm_client = llm_client or LLMClient()
        self.tools = tools or register_tools()
        self.system_message = system_message or CVD_SYSTEM_PROMPT
        self.history: List[Message] = []
        
        # Add system message to history
        self._add_message("system", self.system_message)
    
    def _add_message(self, role: str, content: str, 
                    tool_calls: Optional[List[Dict[str, Any]]] = None,
                    tool_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: Message role
            content: Message content
            tool_calls: Optional tool calls information
            tool_results: Optional tool execution results
        """
        self.history.append(Message(
            role=role,
            content=content,
            timestamp=time.time(),
            tool_calls=tool_calls,
            tool_results=tool_results
        ))
    
    def get_formatted_history(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get the chat history formatted for LLM API.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            Formatted chat history
        """
        formatted_history = []
        
        for message in self.history:
            if message.role == "system" and not include_system:
                continue
                
            formatted_history.append({
                "role": message.role,
                "content": message.content
            })
        
        return formatted_history
    
    def get_conversation_context(self) -> str:
        """
        Get the conversation context as a formatted string.
        
        Returns:
            Formatted conversation context
        """
        context = []
        
        for message in self.history:
            if message.role == "system":
                continue
                
            prefix = f"{message.role.capitalize()}: "
            content = message.content
            
            context.append(f"{prefix}{content}")
        
        return "\n\n".join(context)
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the chat history.
        
        Args:
            content: Message content
        """
        self._add_message("user", content)
    
    def add_assistant_message(self, content: str, 
                             tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add an assistant message to the chat history.
        
        Args:
            content: Message content
            tool_calls: Optional tool calls information
        """
        self._add_message("assistant", content, tool_calls=tool_calls)
    
    def add_tool_result(self, tool_name: str, results: Dict[str, Any]) -> None:
        """
        Add a tool result to the chat history.
        
        Args:
            tool_name: Name of the tool
            results: Tool execution results
        """
        content = f"Tool '{tool_name}' executed with results: {json.dumps(results, indent=2)}"
        self._add_message("tool", content, tool_results=results)
    
    def generate_response(self, 
                         use_tools: bool = True, 
                         temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response from the LLM based on the chat history.
        
        Args:
            use_tools: Whether to enable tool use
            temperature: Temperature for response generation
            
        Returns:
            Dictionary containing response and optional tool calls
        """
        # Get the last user message
        last_user_message = None
        for message in reversed(self.history):
            if message.role == "user":
                last_user_message = message.content
                break
        
        if not last_user_message:
            return {"error": "No user message found in history"}
        
        # Get conversation context
        context = self.get_conversation_context()
        
        # Prepare tools
        tools = None
        if use_tools and self.llm_client.provider == "openai":
            tools = self.tools.get_openai_tools()
        
        # Generate response
        response = self.llm_client.generate_response(
            prompt=last_user_message,
            system_message=self.system_message,
            temperature=temperature,
            tools=tools
        )
        
        # TODO: Parse tool calls from response if applicable
        # This would need to be customized for each LLM provider
        
        # Add response to history
        self.add_assistant_message(response)
        
        return {"response": response, "tool_calls": None}
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool and add the result to history.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution results
        """
        try:
            results = self.tools.call_tool(tool_name, **kwargs)
            
            # Add tool result to history
            self.add_tool_result(tool_name, results)
            
            return results
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Error executing tool '{tool_name}': {str(e)}"
            }
            self.add_tool_result(tool_name, error_result)
            return error_result
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear the chat history.
        
        Args:
            keep_system: Whether to keep the system message
        """
        if keep_system:
            system_messages = [msg for msg in self.history if msg.role == "system"]
            self.history = system_messages
        else:
            self.history = []
            # Re-add system message
            self._add_message("system", self.system_message)