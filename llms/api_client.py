"""
API client for interfacing with LLM providers.
"""
import os
import json
from typing import Dict, List, Optional, Union, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    """Client for interacting with various LLM APIs."""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider name ("openai", "anthropic", "huggingface")
            model: Specific model to use (defaults to provider's recommended model)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = self._get_api_key()
        
        # Set default models per provider if not specified
        if not self.model:
            self.model = self._get_default_model()
    
    def _get_api_key(self) -> str:
        """Get the API key for the selected provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return api_key
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return api_key
        elif self.provider == "huggingface":
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HuggingFace API key not found. Set HUGGINGFACE_API_KEY environment variable.")
            return api_key
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_default_model(self) -> str:
        """Get the default model for the selected provider."""
        if self.provider == "openai":
            return "gpt-4"
        elif self.provider == "anthropic":
            return "claude-3-opus-20240229"
        elif self.provider == "huggingface":
            return "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(self, 
                        prompt: str,
                        system_message: Optional[str] = None,
                        temperature: float = 0.7,
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt text
            system_message: Optional system instructions
            temperature: Sampling temperature (0.0 to 1.0)
            tools: List of tool definitions for function calling
            tool_choice: Optional tool to force usage of
            
        Returns:
            Generated text response
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_message, temperature, tools, tool_choice)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_message, temperature)
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, system_message, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(self, 
                        prompt: str, 
                        system_message: Optional[str] = None,
                        temperature: float = 0.7,
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[str] = None) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Prepare additional parameters for tools if provided
            kwargs = {"temperature": temperature}
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response from OpenAI: {str(e)}"
    
    def _generate_anthropic(self, 
                           prompt: str, 
                           system_message: Optional[str] = None,
                           temperature: float = 0.7) -> str:
        """Generate response using Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            system = system_message if system_message else ""
            
            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=temperature,
                system=system,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating response from Anthropic: {str(e)}"
    
    def _generate_huggingface(self, 
                             prompt: str, 
                             system_message: Optional[str] = None,
                             temperature: float = 0.7) -> str:
        """Generate response using HuggingFace API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format the prompt based on whether a system message is provided
            formatted_prompt = prompt
            if system_message:
                formatted_prompt = f"{system_message}\n\n{prompt}"
            
            data = {
                "inputs": formatted_prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 500,
                    "return_full_text": False
                }
            }
            
            api_url = f"https://api-inference.huggingface.co/models/{self.model}"
            response = requests.post(api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            else:
                return f"Error from HuggingFace API: {response.text}"
                
        except Exception as e:
            return f"Error generating response from HuggingFace: {str(e)}"