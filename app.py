import os

import rag_engine
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def _get_api_key() -> str | None:
    session_key = st.session_state.get("GOOGLE_API_KEY")
    return session_key or os.getenv("GOOGLE_API_KEY")


def _is_likely_valid_api_key(api_key: str | None) -> bool:
    if not api_key:
        return False
    api_key = api_key.strip()
    return len(api_key) > 20 and " " not in api_key


def _initialise_state() -> None:
    metadata = rag_engine.load_metadata()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("company_name", metadata.get("company_name", ""))
    st.session_state.setdefault("candidate_company_name", "")
    st.session_state.setdefault("feedback_status", "")
    st.session_state.setdefault("share_text", "")


def _company_selector() -> None:
    metadata = rag_engine.load_metadata()
    companies = metadata.get("known_companies", [])
    current_company = st.session_state.get("company_name", "")

    options = [""] + [company for company in companies if company]
    if current_company and current_company not in options:
        options.append(current_company)

    if len(options) > 1:
        selected = st.selectbox(
            "Company",
            options,
            index=options.index(current_company) if current_company in options else 0,
            format_func=lambda value: value or "Type or process documents to autofill",
            help="This list is filled automatically from processed documents.",
        )
        if selected:
            st.session_state["company_name"] = selected

    company_name = st.text_input(
        "Company name",
        value=st.session_state.get("company_name", ""),
        placeholder="Auto-filled after document processing",
        help="You can also type the company name manually.",
    )
    st.session_state["company_name"] = company_name.strip()

    detected = st.session_state.get("candidate_company_name", "")
    if detected and detected != st.session_state["company_name"]:
        if st.button(f"Use detected company: {detected}", use_container_width=True):
            st.session_state["company_name"] = detected
            st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Company")
        _company_selector()

        st.divider()
        st.subheader("Knowledge Base")
        docs = st.file_uploader(
            "Upload company documents",
            type=["pdf", "txt", "md", "html", "htm", "docx"],
            accept_multiple_files=True,
        )

        if docs:
            detected_name = rag_engine.detect_company_name(docs)
            if detected_name:
                st.session_state["candidate_company_name"] = detected_name

        if st.button("Process Documents", use_container_width=True):
            if not docs:
                st.error("Upload at least one document.")
                return

            api_key = _get_api_key()
            try:
                with st.spinner("Processing documents and building the knowledge base..."):
                    documents = rag_engine.build_documents(docs)
                    detected_name = rag_engine.detect_company_name_from_documents(documents)
                    if detected_name:
                        st.session_state["candidate_company_name"] = detected_name
                        st.session_state["company_name"] = detected_name

                    if not st.session_state["company_name"]:
                        st.error("Could not detect a company name. Enter it manually, then process again.")
                        return

                    chunks = rag_engine.split_documents(documents)
                    chunk_count = rag_engine.build_vector_store(
                        chunks,
                        api_key=api_key,
                        company_name=st.session_state["company_name"],
                    )
            except Exception as exc:
                st.error(f"Failed to build the knowledge base: {exc}")
                return

            mode = "Gemini embeddings" if _is_likely_valid_api_key(api_key) else "local fallback search"
            st.success(
                f"Knowledge base built with {chunk_count} searchable chunks for "
                f"{st.session_state['company_name']} using {mode}."
            )

        if rag_engine.has_index():
            status = rag_engine.index_status()
            st.caption(f"Saved knowledge base found: {status['local_chunks']} chunks.")
        else:
            st.caption("No saved knowledge base yet.")


def _render_messages() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_feedback() -> None:
    if not st.session_state["messages"]:
        return

    last_answer = next(
        (
            message["content"]
            for message in reversed(st.session_state["messages"])
            if message["role"] == "assistant"
        ),
        "",
    )
    if not last_answer:
        return

    st.caption("Feedback")
    like_col, dislike_col, save_col, share_col = st.columns(4)

    with like_col:
        if st.button("👍", help="Like", use_container_width=True):
            rag_engine.save_feedback_action(
                action="like",
                company_name=st.session_state.get("company_name", ""),
                messages=st.session_state["messages"][-4:],
            )
            st.session_state["feedback_status"] = "Liked"
            st.rerun()

    with dislike_col:
        if st.button("👎", help="Dislike", use_container_width=True):
            rag_engine.save_feedback_action(
                action="dislike",
                company_name=st.session_state.get("company_name", ""),
                messages=st.session_state["messages"][-4:],
            )
            st.session_state["feedback_status"] = "Disliked"
            st.rerun()

    with save_col:
        if st.button("🔖", help="Save", use_container_width=True):
            rag_engine.save_answer(
                answer=last_answer,
                company_name=st.session_state.get("company_name", ""),
                messages=st.session_state["messages"][-4:],
            )
            st.session_state["feedback_status"] = "Saved"
            st.rerun()

    with share_col:
        if st.button("↗", help="Share", use_container_width=True):
            rag_engine.save_feedback_action(
                action="share",
                company_name=st.session_state.get("company_name", ""),
                messages=st.session_state["messages"][-4:],
            )
            st.session_state["share_text"] = last_answer
            st.session_state["feedback_status"] = "Ready to share"
            st.rerun()

    if st.session_state["feedback_status"]:
        st.success(st.session_state["feedback_status"])

    if st.session_state["share_text"]:
        st.text_area("Share text", value=st.session_state["share_text"], height=100)


def main() -> None:
    st.set_page_config(page_title="Company RAG Chatbot", page_icon=":office:", layout="wide")
    _initialise_state()

    st.title("Company RAG Chatbot")
    st.caption("Ask questions using your processed company documents.")

    _render_sidebar()
    _render_messages()

    if not _is_likely_valid_api_key(_get_api_key()):
        st.info("The app is running in local fallback mode. Add `GOOGLE_API_KEY` in the server `.env` file to enable Gemini.")

    prompt = st.chat_input("Ask about the uploaded company documents")
    if not prompt:
        _render_feedback()
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["feedback_status"] = ""
    st.session_state["share_text"] = ""
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not rag_engine.has_index():
            answer = "No document index found yet. Upload documents and click `Process Documents` first."
        else:
            try:
                with st.spinner("Searching the company knowledge base..."):
                    result = rag_engine.answer_question(
                        prompt,
                        api_key=_get_api_key(),
                        company_name=st.session_state.get("company_name") or "the company",
                    )
                answer = result["answer"]
            except Exception as exc:
                answer = f"Failed to generate an answer: {exc}"

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    _render_feedback()


if __name__ == "__main__":
    main()
