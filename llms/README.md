# LLM Integration for CVD-PINN

This module provides integration between Large Language Models (LLMs) and the CVD-PINN framework.

## Overview

The LLM integration enables:

1. Natural language interaction with the CVD-PINN simulation
2. Automatic parameter configuration through conversation
3. Result interpretation and explanation
4. Code generation for custom model components
5. Knowledge-enhanced simulation guidance

## Components

- **API Integration**: Connectors to various LLM providers (OpenAI, Anthropic, HuggingFace)
- **Prompt Engineering**: Specialized prompts for CVD domain tasks
- **Tool Use**: Function definitions for LLM tool use capabilities
- **Context Management**: Handling conversation history and simulation context
- **Knowledge Retrieval**: Access to CVD-specific literature and reference data

## Configuration

Set up your LLM integration by configuring the API keys in the `.env` file or through the Streamlit interface.