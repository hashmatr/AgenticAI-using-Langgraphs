import json
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


DAY6_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = DAY6_DIR / "Chatbot.ipynb"


def get_text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content) if content else ""


@st.cache_resource
def create_chatbot():
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    namespace = {"__name__": "__main__"}
    previous_directory = Path.cwd()

    try:
        os.chdir(DAY6_DIR)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell.get("source", []))
            if not source.strip() or "while True:" in source:
                continue

            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_directory)

    if "workflow" not in namespace:
        namespace["workflow"] = namespace["graph"].compile(
            checkpointer=namespace["checkpointer"]
        )

    return namespace["workflow"]


st.set_page_config(page_title="LangGraph Chatbot", page_icon=":speech_balloon:")
st.title("LangGraph Chatbot")

workflow = create_chatbot()
if workflow is None:
    st.error(
        "Missing Groq API key. Add `groq_api` or `GROQ_API_KEY` to "
        f"`{DAY6_DIR / '.env'}`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-chat"

with st.sidebar:
    st.header("Chat")
    st.caption("Conversation memory is maintained for this browser session.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit-chat-{os.urandom(4).hex()}"
        st.rerun()

for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

prompt = st.chat_input("Type your message...")
if prompt:
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }
            response_placeholder = st.empty()
            streamed_response = ""

            for chunk, _metadata in workflow.stream(
                {"messages": [user_message]},
                config=config,
                stream_mode="messages",
            ):
                text_chunk = get_text_content(
                    getattr(chunk, "content", "")
                )
                if text_chunk:
                    streamed_response += text_chunk
                    response_placeholder.markdown(streamed_response + "▌")

            if streamed_response:
                response_placeholder.markdown(streamed_response)
                st.session_state.messages.append(
                    AIMessage(content=streamed_response)
                )
            else:
                result = workflow.invoke(
                    {"messages": [user_message]},
                    config=config,
                )
                assistant_message = result["messages"][-1]
                response_placeholder.markdown(assistant_message.content)
                st.session_state.messages.append(
                    AIMessage(content=assistant_message.content)
                )
        except Exception as exc:
            st.error(f"Unable to generate a response: {exc}")