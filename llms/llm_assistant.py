"""
LLM assistant page for the Streamlit application.
"""
import streamlit as st
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Any

# Add the parent directory to sys.path to import the package
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import LLM integration modules
from llms.api_client import LLMClient
from llms.prompt_templates import list_available_templates, get_prompt_template
from llms.tools import CVDPINNTools, register_tools
from llms.chat_session import ChatSession


def show_llm_assistant_page():
    """Display the LLM assistant page content."""
    st.title("CVD-PINN AI Assistant")
    
    st.markdown(
        """
        This AI assistant helps you design, configure, and interpret CVD simulations.
        It can answer questions about CVD processes, suggest model parameters, interpret results,
        and even generate custom code for your specific needs.
        """
    )
    
    # Initialize session state for chat
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Prompt Templates", "Configuration"])
    
    # Chat tab
    with tab1:
        _show_chat_interface()
    
    # Prompt templates tab
    with tab2:
        _show_prompt_templates()
    
    # Configuration tab
    with tab3:
        _show_llm_configuration()


def _show_chat_interface():
    """Display the chat interface."""
    # Initialize or get chat session
    if st.session_state.chat_session is None:
        # Default to OpenAI if keys are configured
        provider = st.session_state.get("llm_provider", "openai")
        model = st.session_state.get("llm_model", None)
        
        try:
            llm_client = LLMClient(provider=provider, model=model)
            st.session_state.chat_session = ChatSession(llm_client=llm_client)
            st.session_state.chat_initialized = True
        except ValueError as e:
            st.error(f"Error initializing chat: {str(e)}")
            st.info("Please configure your API keys in the Configuration tab.")
            st.session_state.chat_initialized = False
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        elif message["role"] == "tool":
            # Format tool results for better display
            with st.chat_message("system"):
                st.info(message["content"])
    
    # Chat input
    if st.session_state.get("chat_initialized", False):
        if prompt := st.chat_input("Ask me about CVD simulations..."):
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Add to session and generate response
            st.session_state.chat_session.add_user_message(prompt)
            
            with st.spinner("Thinking..."):
                response_data = st.session_state.chat_session.generate_response()
                
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    response = response_data["response"]
                    
                    # Check if response contains tool calls (simplified parsing)
                    tool_calls = []
                    if "I'll use the" in response and "tool" in response:
                        # This is a simplified way to detect potential tool usage
                        # In a real implementation, you would parse OpenAI's function call format
                        st.warning("The assistant wants to use tools, but automatic tool execution is not implemented in this demo.")
                    
                    # Display assistant response
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
        
        # Button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            if st.session_state.chat_session:
                st.session_state.chat_session.clear_history()
            st.rerun()


def _show_prompt_templates():
    """Display available prompt templates."""
    st.header("Prompt Templates")
    
    st.markdown(
        """
        These templates can help you formulate effective prompts for common CVD-PINN tasks.
        Select a template, fill in the parameters, and copy it to the chat.
        """
    )
    
    # Get available templates
    templates = list_available_templates()
    
    # Template selection
    template_names = [t["name"] for t in templates]
    template_descriptions = {t["name"]: t["description"] for t in templates}
    
    selected_template = st.selectbox(
        "Select a template",
        options=template_names,
        format_func=lambda x: f"{x} - {template_descriptions[x]}"
    )
    
    if selected_template:
        # Get the template
        template = get_prompt_template(selected_template)
        
        st.subheader(f"Template: {selected_template}")
        st.info(template_descriptions[selected_template])
        
        # Display the raw template
        with st.expander("View Template"):
            st.code(template.template)
        
        # Parse template fields
        import re
        fields = re.findall(r'\{([^}]+)\}', template.template)
        fields = list(set(fields))  # Remove duplicates
        
        # Create input fields for each template parameter
        st.subheader("Fill Template Parameters")
        param_values = {}
        
        for field in fields:
            param_values[field] = st.text_area(f"{field.replace('_', ' ').title()}", key=f"param_{field}")
        
        # Preview button
        if st.button("Preview Filled Template"):
            # Format the template with provided values
            try:
                filled_template = template.format(**param_values)
                st.subheader("Preview")
                st.write(filled_template)
                
                # Add a button to send to chat
                if st.button("Send to Chat"):
                    if "chat_messages" in st.session_state:
                        st.session_state.chat_messages.append({"role": "user", "content": filled_template})
                        if st.session_state.chat_session:
                            st.session_state.chat_session.add_user_message(filled_template)
                        st.rerun()
                    else:
                        st.error("Chat is not initialized.")
            except KeyError as e:
                st.error(f"Missing template parameter: {str(e)}")


def _show_llm_configuration():
    """Display LLM configuration options."""
    st.header("LLM Configuration")
    
    # Provider selection
    provider_options = ["openai", "anthropic", "huggingface"]
    selected_provider = st.selectbox(
        "Select LLM Provider",
        options=provider_options,
        index=provider_options.index(st.session_state.get("llm_provider", "openai"))
    )
    
    # Model options based on provider
    if selected_provider == "openai":
        model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif selected_provider == "anthropic":
        model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    else:  # huggingface
        model_options = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-70b-chat-hf"]
    
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=0
    )
    
    # API key input
    api_key = st.text_input(
        f"{selected_provider.title()} API Key",
        type="password",
        value=os.environ.get(f"{selected_provider.upper()}_API_KEY", "")
    )
    
    # Save configuration
    if st.button("Save Configuration"):
        # Save provider and model in session state
        st.session_state.llm_provider = selected_provider
        st.session_state.llm_model = selected_model
        
        # Save API key as environment variable
        os.environ[f"{selected_provider.upper()}_API_KEY"] = api_key
        
        # Reset chat session to apply new settings
        st.session_state.chat_session = None
        
        st.success("Configuration saved! Chat will use the new settings.")
        st.rerun()
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        use_tools = st.checkbox(
            "Enable Tools (Function Calling)",
            value=True,
            help="Enable tools for the assistant to perform actions"
        )
        
        st.session_state.temperature = temperature
        st.session_state.use_tools = use_tools
        
        st.info("Note: Tool use is currently only fully supported with OpenAI models.")