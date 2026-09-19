import streamlit as st
from pathlib import Path
import textwrap

from rag_engine import build_rag_index, ask_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Placement Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background: #f8fafc;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}


/* ==========================================================
   GLOBAL TEXT
   ========================================================== */

h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

p {
    color: #334155;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

.sidebar-brand {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 16px;
    color: #cbd5e1 !important;
    margin-bottom: 25px;
}

.sidebar-heading {
    color: #94a3b8 !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-top: 25px;
    margin-bottom: 12px;
}

.sidebar-item {
    color: #e2e8f0 !important;
    padding: 11px 14px;
    border-radius: 9px;
    margin-bottom: 6px;
    font-size: 15px;
}

.sidebar-item.active {
    background: #1e293b;
    border: 1px solid #334155;
}

.sidebar-divider {
    height: 1px;
    background: #334155;
    margin: 25px 0;
}

.sidebar-status {
    background: #064e3b;
    border: 1px solid #047857;
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
}

.sidebar-status-title {
    color: #ecfdf5 !important;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 8px;
}

.sidebar-status-item {
    color: #a7f3d0 !important;
    font-size: 13px;
    margin-top: 5px;
}


/* ==========================================================
   HEADER
   ========================================================== */

.app-title {
    font-size: 42px;
    font-weight: 750;
    line-height: 1.15;
    color: #0f172a !important;
    margin-bottom: 8px;
}

.app-subtitle {
    font-size: 17px;
    color: #64748b !important;
    margin-bottom: 25px;
}


/* ==========================================================
   STATUS MESSAGE
   ========================================================== */

.status-box {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 12px;
    padding: 15px 18px;
    margin-bottom: 32px;
}

.status-box-text {
    color: #065f46 !important;
    font-size: 14px;
    font-weight: 600;
}


/* ==========================================================
   SECTION TITLE
   ========================================================== */

.section-title {
    color: #0f172a !important;
    font-size: 22px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 16px;
}


/* ==========================================================
   FEATURE CARDS
   ========================================================== */

.feature-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    min-height: 175px;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 13px;
}

.feature-title {
    color: #0f172a !important;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 9px;
}

.feature-description {
    color: #64748b !important;
    font-size: 14px;
    line-height: 1.6;
}


/* ==========================================================
   FILE UPLOADER - WHITE PROFESSIONAL THEME
   ========================================================== */

/* Outer uploader */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Uploader drop area */
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

/* Text inside uploader */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #0f172a !important;
}

/* Upload button */
[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 8px !important;
}

/* Upload button text */
[data-testid="stFileUploaderDropzone"] button span {
    color: #0f172a !important;
}

/* Upload button icon */
[data-testid="stFileUploaderDropzone"] button svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* ==========================================================
   UPLOADED FILE CHIP
   ========================================================== */

/* File chip */
[data-testid="stFileUploaderFile"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}

/* Everything inside file chip */
[data-testid="stFileUploaderFile"] * {
    background-color: transparent !important;
    color: #0f172a !important;
}

/* Filename */
[data-testid="stFileUploaderFile"] span {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* File size */
[data-testid="stFileUploaderFile"] small {
    color: #64748b !important;
}

/* File icon */
[data-testid="stFileUploaderFile"] svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* Remove X button */
[data-testid="stFileUploaderFile"] button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: none !important;
}

/* X icon */
[data-testid="stFileUploaderFile"] button svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* Plus / add-file button */
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
}

/* Hover */
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #f8fafc !important;
    color: #0f172a !important;
}
/* ==========================================================
   CHAT AREA
   ========================================================== */

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #eef2ff;
    border-radius: 14px;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}


/* Force ALL assistant text to dark */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #1e293b !important;
}


/* Assistant answer */
.answer-text {
    color: #1e293b !important;
    font-size: 16px;
    line-height: 1.7;
}


/* ==========================================================
   SOURCE SECTION
   ========================================================== */

.sources-title {
    color: #334155 !important;
    font-size: 14px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 8px;
}

.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    padding: 10px 14px;
    margin-top: 7px;
}

.source-name {
    color: #475569 !important;
    font-size: 13px;
    font-weight: 600;
}


/* ==========================================================
   CHAT INPUT
   ========================================================== */

[data-testid="stChatInput"] {
    background: #ffffff;
}

[data-testid="stChatInput"] textarea {
    color: #0f172a !important;
    background: #ffffff !important;
    caret-color: #2563eb !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton button {
    border-radius: 9px;
}


/* ==========================================================
   SPINNER
   ========================================================== */

.stSpinner > div {
    color: #2563eb !important;
}
/* ==========================================================
   TEXT SELECTION
   ========================================================== */

/* Selected text across the application */
/* ==========================================================
   DOCUMENT FILENAME
   ========================================================== */

/* Normal filename */
.document-name {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Selected filename */
.document-name::selection {
    background: #dbeafe !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🤖 AI Placement</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item active">💬 &nbsp; Ask Documents</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">📄 &nbsp; Documents</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">📊 &nbsp; Placement Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">⚙️ &nbsp; Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">SYSTEM STATUS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-status">'
        '<div class="sidebar-status-title">● AI Engine Ready</div>'
        '<div class="sidebar-status-item">Gemini + RAG</div>'
        '<div class="sidebar-status-item">Local Embeddings</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="app-title">AI Placement Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Your intelligent assistant for understanding placement documents.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status-box">'
    '<div class="status-box-text">'
    '✓ AI system is ready. Upload your placement documents '
    'and start asking questions.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

st.markdown(
    '<div class="section-title">What can I help you with?</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">📄</div>'
        '<div class="feature-title">Analyze Documents</div>'
        '<div class="feature-description">'
        'Understand placement notifications, job descriptions '
        'and company documents.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">🔍</div>'
        '<div class="feature-title">Find Information</div>'
        '<div class="feature-description">'
        'Find eligibility, salary, skills, locations '
        'and selection details.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">🤖</div>'
        '<div class="feature-title">AI-Powered Answers</div>'
        '<div class="feature-description">'
        'Get answers grounded in your uploaded placement '
        'documents using Retrieval-Augmented Generation.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Placement Documents</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload placement notifications, job descriptions or company documents."
)


# ============================================================
# DETERMINE ACTIVE DOCUMENTS
# ============================================================

if uploaded_files:

    upload_folder = Path("uploaded_documents")

    upload_folder.mkdir(
        exist_ok=True
    )

    uploaded_paths = []

    for uploaded_file in uploaded_files:

        file_path = (
            upload_folder /
            uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        uploaded_paths.append(
            file_path
        )

    active_pdf_files = uploaded_paths

else:

    default_folder = Path("documents")

    active_pdf_files = list(
        default_folder.glob("*.pdf")
    )


# ============================================================
# ACTIVE DOCUMENTS
# ============================================================

st.markdown(
    '<div class="section-title">📚 Active Documents</div>',
    unsafe_allow_html=True
)

if active_pdf_files:

    for pdf in active_pdf_files:

        st.markdown(
            '<div class="document-card">'
            f'<span class="document-name">📄 {pdf.name}</span>'
            '</div>',
            unsafe_allow_html=True
        )

else:

    st.warning(
        "No PDF documents available. Please upload a PDF."
    )


# ============================================================
# BUILD RAG INDEX
# ============================================================

if active_pdf_files:

    document_names = tuple(
        sorted(
            pdf.name
            for pdf in active_pdf_files
        )
    )

    previous_names = st.session_state.get(
        "document_names"
    )

    if (
        "rag_index" not in st.session_state
        or previous_names != document_names
    ):

        with st.spinner(
            "Reading documents and preparing the AI..."
        ):

            st.session_state.rag_index = (
                build_rag_index(
                    active_pdf_files
                )
            )

            st.session_state.document_names = (
                document_names
            )


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask Your Documents</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=message["avatar"]
    ):

        if message["role"] == "assistant":

            st.markdown(
                f'<div class="answer-text">'
                f'{message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.write(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your placement documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "avatar": "👤"
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(
            question
        )


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "Searching your documents..."
        ):

            try:

                result = ask_question(
                    question,
                    st.session_state.rag_index
                )

                answer = result["answer"]

                sources = result["sources"]


                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.markdown(
                    f'<div class="answer-text">'
                    f'{answer}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                unique_sources = []

                for source in sources:

                    filename = source["source"]

                    if filename not in unique_sources:

                        unique_sources.append(
                            filename
                        )


                if unique_sources:

                    st.markdown(
                        '<div class="sources-title">'
                        '📚 Sources'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    for filename in unique_sources:

                        st.markdown(
                            '<div class="source-card">'
                            f'<span class="source-name">'
                            f'📄 {filename}'
                            f'</span>'
                            '</div>',
                            unsafe_allow_html=True
                        )


                # ------------------------------------------------
                # SAVE AI MESSAGE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "avatar": "🤖"
                    }
                )


            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(
                    error_message
                )