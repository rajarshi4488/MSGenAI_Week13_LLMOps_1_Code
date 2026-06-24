import streamlit as st
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chain import SupplyChainAssistant
from src.config import Config


# Create instance (cached based on config hash)
# We use a hash of the overrides to invalidate cache when configs change
@st.cache_resource(hash_funcs={dict: lambda d: hash(frozenset(d.items())) if d else 0})
def get_assistant(overrides: dict = None):
    return SupplyChainAssistant(overrides=overrides)


def main():
    st.set_page_config(page_title="DataCo Supply Chain AI", layout="wide")

    st.title("📦 DataCo Supply Chain AI Assistant")
    st.markdown("Supply Chain AI Copilots - Ask questions about order details, shipping modes, risks, and financials.")

    # Sidebar for Configuration
    st.sidebar.header("Configuration")

    # API Configs
    api_key = st.sidebar.text_input(
        "Azure OpenAI API Key",
        value=Config.AZURE_OPENAI_API_KEY if Config.AZURE_OPENAI_API_KEY else "",
        type="password",
    )
    endpoint = st.sidebar.text_input(
        "Azure OpenAI Endpoint",
        value=Config.AZURE_OPENAI_ENDPOINT if Config.AZURE_OPENAI_ENDPOINT else "",
    )
    api_version = st.sidebar.text_input("API Version", value=Config.AZURE_OPENAI_API_VERSION)

    # Model Configs
    st.sidebar.subheader("Model Deployments")
    agent_model = st.sidebar.text_input(
        "Language Model", value=os.getenv("AZURE_DEPLOYMENT", "gpt-5-mini")
    )
    judge_model = st.sidebar.text_input("Judge Model", value=Config.AZURE_JUDGE_DEPLOYMENT)

    # Clear History Button
    if st.sidebar.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

    # Prepare overrides
    overrides = {
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_VERSION": api_version,
        "AZURE_DEPLOYMENT": agent_model,
        "AZURE_JUDGE_DEPLOYMENT": judge_model,
    }

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display message content
            st.markdown(message["content"])
            # Display additional info if present
            # If the message contains code, data, or image, show them in expanders
            if "code" in message:
                with st.expander("View Code"):
                    st.code(message["code"], language="python")
            if "data" in message:
                with st.expander("View Raw Result"):
                    st.text(message["data"])
            if "image" in message:
                st.pyplot(message["image"])

    # User input
    user_query = st.chat_input("Ask about supply chain data...")

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing supply chain data..."):
                assistant = get_assistant(overrides)
                result = assistant.execute_query(user_query)

                response_text = result.get("response", "I encountered an error.")
                code_text = result.get("code", None)
                data_text = result.get("data", None)
                image_obj = result.get("image", None)

                st.markdown(response_text)

                if image_obj:
                    st.pyplot(image_obj)

                if code_text:
                    with st.expander("Code & Debug Info"):
                        st.code(code_text, language="python")
                        if data_text:
                            st.text("Raw Result:")
                            st.text(data_text)

        # Save assistant message
        msg = {"role": "assistant", "content": response_text}
        if code_text:
            msg["code"] = code_text
        if data_text:
            msg["data"] = data_text
        if image_obj:
            msg["image"] = image_obj
        st.session_state.messages.append(msg)


if __name__ == "__main__":
    main()
