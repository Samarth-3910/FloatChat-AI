import streamlit as st
import requests

# ---------------- Page Config ----------------
st.set_page_config(layout="wide", page_title="Ocean Data Chat")

# ---------------- Session State ----------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Default Session": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Default Session"

FLASK_BACKEND_URL = "http://localhost:5000/api/chat"

# ---------------- Sidebar (Session Management) ----------------
with st.sidebar:
    st.title("💬 Chat Sessions")

    session_names = list(st.session_state.chat_sessions.keys())
    selected_session = st.radio(
        "Select a session",
        session_names,
        index=session_names.index(st.session_state.current_session)
    )

    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        st.rerun()

    st.markdown("---")
    if st.button("➕ Create New Session"):
        new_session_name = f"Session {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.success(f"New session '{new_session_name}' created!")

# ---------------- Chat History ----------------
chat_history_container = st.container()
with chat_history_container:
    for message in st.session_state.chat_sessions[st.session_state.current_session]:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                # Backend returns human-readable string, so just render markdown
                st.markdown(content)
        else:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(content)

# ---------------- Send Prompt Function ----------------
def send_prompt():
    user_prompt = st.session_state.chat_input_widget.strip()
    if not user_prompt:
        return

    current_session = st.session_state.current_session
    # Add user message to chat
    st.session_state.chat_sessions[current_session].append(
        {"role": "user", "content": user_prompt}
    )

    with st.spinner("Thinking..."):
        try:
            response = requests.post(FLASK_BACKEND_URL, json={"prompt": user_prompt})
            response.raise_for_status()
            bot_response = response.json().get("response", "No response")
        except Exception as e:
            bot_response = f"❌ Error: {e}"

    # Add bot response to chat
    st.session_state.chat_sessions[current_session].append(
        {"role": "assistant", "content": bot_response}
    )

    # Clear input widget
    st.session_state.chat_input_widget = ""

# ---------------- Chat Input ----------------
st.chat_input(
    "Type your message here...", 
    on_submit=send_prompt, 
    key="chat_input_widget"
)
