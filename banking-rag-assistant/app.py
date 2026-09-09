import streamlit as st

from rag_engine import ask_question


# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="AI Banking Knowledge Assistant",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 AI Banking Knowledge Assistant")

st.caption(
    "Ask questions about banking, payments and fraud. "
    "Answers are grounded in the banking knowledge base."
)


# --------------------------------------------------
# 2. Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 3. Display source citations
# --------------------------------------------------

def display_sources(sources):

    if not sources:
        return

    st.markdown("#### Sources")

    for source in sources:

        if source["type"] == "pdf":

            st.write(
                f"📄 {source['name']} — "
                f"Page {source['page']}"
            )

        elif source["type"] == "web":

            st.markdown(
                f"🌐 [{source['name']}]({source['url']})"
            )

        else:

            st.write(
                f"📚 {source['name']}"
            )


# --------------------------------------------------
# 4. Show previous chat
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            display_sources(
                message["sources"]
            )


# --------------------------------------------------
# 5. User input
# --------------------------------------------------

question = st.chat_input(
    "Ask a banking question..."
)


# --------------------------------------------------
# 6. Process question
# --------------------------------------------------

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


        answer = result["answer"]
        sources = result["sources"]
        retrieval_results = (
            result["retrieval_results"]
        )


        st.markdown(answer)

        display_sources(sources)


        # ------------------------------------------
        # Retrieval details
        # ------------------------------------------

        with st.expander(
            "🔎 View retrieval details"
        ):

            for i, (
                document,
                score
            ) in enumerate(
                retrieval_results,
                start=1
            ):

                source_name = (
                    document.metadata.get(
                        "source_name",
                        "Unknown source"
                    )
                )

                page_number = (
                    document.metadata.get(
                        "page_number"
                    )
                )

                if page_number:
                    location = (
                        f"Page {page_number}"
                    )
                else:
                    location = "Webpage"


                st.write(
                    f"Result {i} | "
                    f"Score: {score:.3f} | "
                    f"{source_name} | "
                    f"{location}"
                )


                snippet = (
                    document.page_content[:300]
                    .replace("\n", " ")
                )

                st.caption(
                    snippet + "..."
                )

                st.divider()


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# --------------------------------------------------
# 7. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses "
        "Retrieval-Augmented Generation (RAG) "
        "to answer banking questions using "
        "selected banking knowledge sources."
    )

    st.markdown(
        """
**Pipeline**

Documents / Webpages  
→ Text Chunking  
→ OpenAI Embeddings  
→ ChromaDB  
→ Semantic Retrieval  
→ LLM  
→ Grounded Answer
"""
    )

    st.caption(
        "This application is for educational "
        "and demonstration purposes."
    )


    if st.button(
        "🗑️ Clear conversation"
    ):

        st.session_state.messages = []

        st.rerun()