
import json
import os
from pathlib import Path
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


DAY6_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = DAY6_DIR / "Chatbot.ipynb"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_text_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return str(content) if content else ""


def generate_thread_id():
    return str(uuid.uuid4())


def load_conversation(thread_id):
    """Load messages from a LangGraph checkpoint."""

    try:
        state = workflow.get_state(
            config={
                "configurable": {"thread_id": thread_id}
            }
        )

        if not state or not state.values:
            return []

        return state.values.get("messages", [])

    except Exception as exc:
        st.warning(f"Could not load conversation: {exc}")
        return []


def add_threads(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():
    """Create a new conversation."""

    thread_id = generate_thread_id()

    st.session_state.thread_id = thread_id
    st.session_state.messages = []

    add_threads(thread_id)

    st.rerun()


def switch_thread(thread_id):
    """Switch to an existing conversation."""

    st.session_state.thread_id = thread_id
    st.session_state.messages = load_conversation(thread_id)

    st.rerun()


# --------------------------------------------------
# Initialize chatbot from Jupyter Notebook
# --------------------------------------------------

@st.cache_resource
def create_chatbot():

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

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

            exec(
                compile(source, str(NOTEBOOK_PATH), "exec"),
                namespace
            )

    finally:
        os.chdir(previous_directory)

    if "workflow" not in namespace:

        if "graph" not in namespace or "checkpointer" not in namespace:
            raise RuntimeError(
                "Could not find workflow or graph/checkpointer "
                "in Chatbot.ipynb."
            )

        namespace["workflow"] = namespace["graph"].compile(
            checkpointer=namespace["checkpointer"]
        )

    return namespace["workflow"]


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("LangGraph Chatbot")


# --------------------------------------------------
# Create workflow
# --------------------------------------------------

try:
    workflow = create_chatbot()

except Exception as exc:
    st.error(f"Unable to initialize chatbot: {exc}")
    st.stop()

if workflow is None:
    st.error(
        "Missing Groq API key. Add `groq_api` or `GROQ_API_KEY` "
        f"to {DAY6_DIR / '.env'}."
    )
    st.stop()


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "messages" not in st.session_state:
    st.session_state.messages = load_conversation(
        st.session_state.thread_id
    )

add_threads(st.session_state.thread_id)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("💬 LangGraph Chatbot")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()

    st.divider()

    st.header("My Conversations")

    if not st.session_state.chat_threads:
        st.caption("No conversations yet.")

    for thread_id in st.session_state.chat_threads:

        if st.button(
            f"Chat {thread_id[:8]}",
            key=f"thread_{thread_id}",
            use_container_width=True
        ):

            if thread_id != st.session_state.thread_id:
                switch_thread(thread_id)

    st.divider()

    st.caption(
        "Conversation memory is maintained using LangGraph checkpoints."
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        reset_chat()


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        role = "user"

    elif isinstance(message, AIMessage):
        role = "assistant"

    else:
        continue

    content = get_text_content(message.content)

    with st.chat_message(role):
        st.markdown(content)


# --------------------------------------------------
# Chat input and response streaming
# --------------------------------------------------

prompt = st.chat_input("Type your message...")

if prompt:

    user_message = HumanMessage(content=prompt)

    # Display user message
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            response_placeholder = st.empty()
            streamed_response = ""

            # Stream response from LangGraph
            for chunk, metadata in workflow.stream(
                {"messages": [user_message]},
                config=config,
                stream_mode="messages"
            ):

                # Only process AI message chunks
                if not isinstance(chunk, AIMessage):
                    continue

                # Ignore tool-call messages
                text_chunk = get_text_content(
                    getattr(chunk, "content", "")
                )

                if text_chunk:

                    streamed_response += text_chunk

                    response_placeholder.markdown(
                        streamed_response + "▌"
                    )

            # Display final response
            if streamed_response:

                response_placeholder.markdown(streamed_response)

            else:
                # Recover the final response from the checkpoint
                state = workflow.get_state(config=config)

                saved_messages = (
                    state.values.get("messages", [])
                    if state and state.values
                    else []
                )

                assistant_messages = [
                    message for message in saved_messages
                    if isinstance(message, AIMessage)
                    and get_text_content(message.content)
                ]

                if assistant_messages:
                    assistant_response = get_text_content(
                        assistant_messages[-1].content
                    )

                    response_placeholder.markdown(
                        assistant_response
                    )

                else:
                    response_placeholder.warning(
                        "The chatbot did not return a text response."
                    )

            # Reload messages from the checkpoint
            # to keep Streamlit and LangGraph synchronized
            st.session_state.messages = load_conversation(
                st.session_state.thread_id
            )

        except Exception as exc:

            st.error(
                f"Unable to generate a response: {exc}"
            )
