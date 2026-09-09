import streamlit as st

from rag_engine import ask_question, APP_VERSION


st.set_page_config(
    page_title="AI Banking Knowledge Assistant",
    page_icon="🏦",
    layout="centered",
)


# ==================================================
# Reset old Streamlit session after code update
# ==================================================

if st.session_state.get("app_version") != APP_VERSION:
    st.session_state.messages = []
    st.session_state.app_version = APP_VERSION


if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# Source display
# ==================================================

def display_sources(sources):

    if not sources:
        return

    st.markdown("### Sources")

    for source in sources:

        source_type = source.get("type")
        name = source.get("name", "Unknown source")

        if source_type == "pdf":

            page = source.get("page")

            st.markdown(
                f"- **{name}** — Page {page}"
            )

        elif source_type == "web":

            url = source.get("url")

            st.markdown(
                f"- [{name}]({url})"
            )

        else:

            st.markdown(
                f"- **{name}**"
            )


# ==================================================
# Retrieval details
# ==================================================

def display_retrieval_details(retrieval_results):

    with st.expander("🔎 View retrieval details"):

        if not retrieval_results:
            st.write("No documents were retrieved.")
            return

        for index, result in enumerate(
            retrieval_results,
            start=1,
        ):

            document, score = result

            source_name = document.metadata.get(
                "source_name",
                "Unknown source",
            )

            page_number = document.metadata.get(
                "page_number"
            )

            url = document.metadata.get("url")

            st.markdown(
                f"### Result {index}"
            )

            st.write(
                f"Relevance score: {score:.3f}"
            )

            st.write(
                f"Source: {source_name}"
            )

            if page_number:

                st.write(
                    f"Page: {page_number}"
                )

            if url:

                st.write(
                    f"URL: {url}"
                )

            st.write("Retrieved text:")

            st.write(
                document.page_content[:700]
            )

            st.divider()


# ==================================================
# Header
# ==================================================

st.title("🏦 AI Banking Knowledge Assistant")

st.caption(
    "Ask questions about banking, payments and fraud. "
    "Answers are grounded in the banking knowledge base."
)


# ==================================================
# Existing conversation
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )

            display_retrieval_details(
                message.get(
                    "retrieval_results",
                    [],
                )
            )


# ==================================================
# New question
# ==================================================

question = st.chat_input(
    "Ask a banking question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching banking knowledge..."
        ):

            result = ask_question(question)

        st.markdown(
            result["answer"]
        )

        display_sources(
            result["sources"]
        )

        display_retrieval_details(
            result["retrieval_results"]
        )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_results": result[
                "retrieval_results"
            ],
        }
    )


# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This project uses Retrieval-Augmented "
        "Generation (RAG) to answer questions using "
        "a banking knowledge base."
    )

    st.markdown(
        """
**Pipeline**

Question  
↓  
OpenAI Embedding  
↓  
Chroma Vector Search  
↓  
Relevant Banking Documents  
↓  
LLM Answer
"""
    )

    st.info(
        "Educational project only. "
        "This application does not provide "
        "financial advice."
    )

    st.caption(
        f"App version: {APP_VERSION}"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()