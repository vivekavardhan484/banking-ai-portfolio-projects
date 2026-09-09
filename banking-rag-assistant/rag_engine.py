from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.6-luna"

TOP_K = 4
RELEVANCE_THRESHOLD = 0.3


# --------------------------------------------------
# Load models and vector store
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model=LLM_MODEL
)


# --------------------------------------------------
# Retrieve documents
# --------------------------------------------------

def retrieve_documents(question):

    results_with_scores = (
        vector_store.similarity_search_with_relevance_scores(
            question,
            k=TOP_K
        )
    )

    relevant_results = [
        document
        for document, score in results_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    return results_with_scores, relevant_results


# --------------------------------------------------
# Format unique sources
# --------------------------------------------------

def get_unique_sources(documents):

    unique_sources = []
    seen = set()

    for document in documents:

        source_name = document.metadata.get(
            "source_name",
            "Unknown source"
        )

        page_number = document.metadata.get("page_number")
        url = document.metadata.get("url")

        if page_number:

            key = (
                source_name,
                page_number
            )

            if key not in seen:

                unique_sources.append(
                    {
                        "type": "pdf",
                        "name": source_name,
                        "page": page_number,
                    }
                )

                seen.add(key)

        elif url:

            key = (
                source_name,
                url
            )

            if key not in seen:

                unique_sources.append(
                    {
                        "type": "web",
                        "name": source_name,
                        "url": url,
                    }
                )

                seen.add(key)

        else:

            key = source_name

            if key not in seen:

                unique_sources.append(
                    {
                        "type": "other",
                        "name": source_name,
                    }
                )

                seen.add(key)

    return unique_sources


# --------------------------------------------------
# Build context
# --------------------------------------------------

def build_context(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# --------------------------------------------------
# Generate grounded answer
# --------------------------------------------------

def generate_answer(question, documents):

    if not documents:

        return (
            "I don't have enough information "
            "in the provided banking documents."
        )

    context = build_context(documents)

    prompt = f"""
You are a banking knowledge assistant.

Answer the user's question using only the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Give a clear and concise answer.
4. If the context does not contain enough information to answer,
   say exactly:

"I don't have enough information in the provided banking documents."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# Complete RAG pipeline
# --------------------------------------------------

def ask_question(question):

    results_with_scores, relevant_results = (
        retrieve_documents(question)
    )

    answer = generate_answer(
        question,
        relevant_results
    )

    sources = get_unique_sources(
        relevant_results
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_results": results_with_scores,
        "relevant_documents": relevant_results,
    }