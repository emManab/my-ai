import streamlit as st
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Manav 🦋",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0d0f14;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 45px;
        padding-bottom: 120px;
    }

    /* Title */
    .title {
        font-size: 48px;
        font-weight: 800;
        color: #f5f5f5;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #8e8e96;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* User message */
    .user-message {
        background: #1b1d24;
        border-radius: 14px;
        padding: 15px 18px;
        margin: 12px 0;
        color: #f1f1f1;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .user-icon {
        background: #ff304f;
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    /* AI message */
    .ai-message {
        padding: 15px 18px;
        margin: 12px 0 25px 0;
        color: #eeeeee;
        font-size: 16px;
        line-height: 1.6;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }

    .ai-icon {
        background: #ff9f1c;
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    .message-text {
        padding-top: 6px;
    }

    /* Thinking */
    .thinking {
        color: #85858d;
        font-size: 14px;
        margin: 15px 0;
    }

    /* Input */
    div[data-testid="stChatInput"] {
        background: #202127;
        border: 1px solid #ff304f;
        border-radius: 12px;
    }

    div[data-testid="stChatInput"] textarea {
        color: white;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD BACKEND
# ============================================================

@st.cache_resource(show_spinner="Loading Manav AI...")
def load_backend():

    import src.chat as backend

    return backend


backend = load_backend()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">Manav 🦋</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Your personal texting assistant</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Manav AI 🦋")

    st.write(
        "Your personal texting assistant."
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        # Clear backend conversation memory too
        backend.conversation_history.clear()

        st.rerun()

    st.divider()

    st.caption(
        f"RAG memories: {backend.collection.count():,}"
    )

    st.caption(
        "BGE-M3 + ChromaDB + Gemini"
    )


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                <div class="user-icon">😎</div>
                <div>{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        # Convert newlines to <br>
        formatted_text = (
            message["content"]
            .replace("\n", "<br>")
        )

        st.markdown(
            f"""
            <div class="ai-message">
                <div class="ai-icon">🦋</div>
                <div class="message-text">
                    {formatted_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Type a message..."
)


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Add user message to UI
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Add user message to backend memory
    # --------------------------------------------------------

    backend.add_to_history(
        "Person",
        user_input
    )

    # --------------------------------------------------------
    # Thinking indicator
    # --------------------------------------------------------

    with st.spinner("Thinking..."):

        try:

            response = backend.generate_response(
                user_input
            )

        except Exception as error:

            response = (
                "bro something went wrong 😭\n\n"
                f"`{error}`"
            )

            # Remove user message from backend
            if backend.conversation_history:

                backend.conversation_history.pop()

    # --------------------------------------------------------
    # Add AI response to backend memory
    # --------------------------------------------------------

    backend.add_to_history(
        "Manab",
        response
    )

    # --------------------------------------------------------
    # Add AI response to UI
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    # --------------------------------------------------------
    # Refresh
    # --------------------------------------------------------

    st.rerun()