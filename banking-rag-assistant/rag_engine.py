from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


# ==================================================
# Configuration
# ==================================================

APP_VERSION = "2026-09-09-v3"

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.6-luna"

TOP_K = 4
RELEVANCE_THRESHOLD = 0.3

FALLBACK_ANSWER = (
    "I don't have enough information in the provided banking documents."
)


# ==================================================
# Models
# ==================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

vector_store = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
)


# ==================================================
# Retrieval
# ==================================================

def retrieve_documents(question):
    results_with_scores = (
        vector_store.similarity_search_with_relevance_scores(
            question,
            k=TOP_K,
        )
    )

    relevant_documents = [
        document
        for document, score in results_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    return results_with_scores, relevant_documents


# ==================================================
# Context builder
# ==================================================

def build_context(documents):
    sections = []

    for index, document in enumerate(documents, start=1):

        source_name = document.metadata.get(
            "source_name",
            "Unknown source",
        )

        page_number = document.metadata.get("page_number")
        url = document.metadata.get("url")

        if page_number:
            location = f"{source_name}, page {page_number}"
        elif url:
            location = f"{source_name}, {url}"
        else:
            location = source_name

        sections.append(
            f"""DOCUMENT {index}
SOURCE: {location}

{document.page_content}"""
        )

    return "\n\n---\n\n".join(sections)


# ==================================================
# Answer generation
# ==================================================

def generate_answer(question, documents):

    # If retrieval found nothing above the threshold,
    # don't waste an LLM call.
    if not documents:
        return FALLBACK_ANSWER

    context = build_context(documents)

    prompt = f"""
You are an AI Banking Knowledge Assistant.

Answer the user's question using ONLY the retrieved
banking context supplied below.

Rules:

1. Use only the retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the retrieved passages contain information relevant
   to the question, answer using that information.
5. The context does NOT need to contain an exact sentence
   matching the user's question.
6. Combine useful information from multiple retrieved
   passages when appropriate.
7. Give a concise, clear answer.
8. Do not invent source names or page numbers.
9. Only return the fallback response when the retrieved
   context genuinely contains no useful information for
   answering the question.

Fallback response:

"{FALLBACK_ANSWER}"

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    if not answer:
        return FALLBACK_ANSWER

    return answer


# ==================================================
# Sources
# ==================================================

def get_unique_sources(documents):
    unique_sources = []
    seen = set()

    for document in documents:

        source_name = document.metadata.get(
            "source_name",
            "Unknown source",
        )

        page_number = document.metadata.get("page_number")
        url = document.metadata.get("url")

        if page_number:

            key = (source_name, page_number)

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

            key = (source_name, url)

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


# ==================================================
# Main RAG pipeline
# ==================================================

def ask_question(question):

    results_with_scores, relevant_documents = (
        retrieve_documents(question)
    )

    answer = generate_answer(
        question,
        relevant_documents,
    )

    sources = get_unique_sources(
        relevant_documents
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_results": results_with_scores,
        "relevant_documents": relevant_documents,
        "app_version": APP_VERSION,
    }