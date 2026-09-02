import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Synapse AI | RAG Workspace",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Synapse AI - Document Assistant</p>', unsafe_allow_html=True)
st.caption("Upload documents or codebase archives to analyze, query, and listen to context-aware answers.")

with st.sidebar:
    st.header("⚙️ Document Hub")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .docx, .zip, .py, .txt)",
        accept_multiple_files=True,
        type=["pdf", "docx", "zip", "py", "txt", "md"]
    )

    if st.button("📤 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please attach at least one file first.")
        else:
            with st.spinner("Processing and vectorizing documents..."):
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
                ]
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
                    if res.status_code == 200:
                        st.success(f"Indexed successfully! Total Chunks: {res.json().get('chunks_indexed')}")
                    else:
                        st.error(f"Error: {res.json().get('detail', 'Failed to upload')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    st.divider()
    allow_web = st.toggle("🌐 Enable Web Search Fallback", value=False)
    enable_audio = st.toggle("🔊 Auto-generate Audio Response", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/mpeg")

if user_prompt := st.chat_input("Ask anything about your documents or repository..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                chat_res = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": user_prompt, "allow_web_search": allow_web}
                )

                if chat_res.status_code == 200:
                    data = chat_res.json()

                    if data.get("status") == "REQUIRES_PERMISSION":
                        answer = data.get("message")
                        st.info(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        answer = data.get("answer")
                        source = data.get("source")
                        st.markdown(f"{answer}\n\n*Source: `{source}`*")

                        msg_data = {"role": "assistant", "content": answer}

                        if enable_audio:
                            tts_res = requests.post(f"{BACKEND_URL}/tts", data={"text": answer})
                            if tts_res.status_code == 200:
                                audio_bytes = tts_res.content
                                st.audio(audio_bytes, format="audio/mpeg")
                                msg_data["audio"] = audio_bytes

                        st.session_state.messages.append(msg_data)
                else:
                    st.error(chat_res.json().get("detail", "Error processing request."))
            except Exception as e:
                st.error(f"Failed to reach FastAPI backend: {e}")