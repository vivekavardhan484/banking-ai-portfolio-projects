from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------------------------
# 2. Connect to existing ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)


# --------------------------------------------------
# 3. User question
# --------------------------------------------------

question = "What problems can customers have understanding retail banking products?"

# --------------------------------------------------
# 4. Retrieve relevant chunks + relevance scores
# --------------------------------------------------

results_with_scores = vector_store.similarity_search_with_relevance_scores(
    question,
    k=4
)


# --------------------------------------------------
# 5. Display retrieval results
# --------------------------------------------------

print("\n--- RETRIEVAL SCORES ---")

for i, (document, score) in enumerate(results_with_scores, start=1):

    source_name = document.metadata.get(
        "source_name",
        "Unknown source"
    )

    page_number = document.metadata.get("page_number")

    if page_number:
        location = f"Page {page_number}"
    else:
        location = "Webpage"

    print(
        f"Result {i} | Score: {score:.3f} | "
        f"{source_name} | {location}"
    )


# --------------------------------------------------
# 6. Filter weak retrieval results
# --------------------------------------------------

# 0.3 is provisional.
# We will calibrate this properly during evaluation.
relevant_results = [
    document
    for document, score in results_with_scores
    if score >= 0.3
]


# --------------------------------------------------
# 7. Stop if nothing relevant was found
# --------------------------------------------------

if not relevant_results:

    print("\n--- QUESTION ---")
    print(question)

    print("\n--- AI ANSWER ---")
    print(
        "I don't have enough information "
        "in the provided banking documents."
    )

    raise SystemExit


# --------------------------------------------------
# 8. Build context for the LLM
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in relevant_results
)


# --------------------------------------------------
# 9. Create LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-5.6-luna"
)


# --------------------------------------------------
# 10. Grounded RAG prompt
# --------------------------------------------------

prompt = f"""
You are a banking knowledge assistant.

Answer the user's question using only the provided context.

Do not use outside knowledge.

If the context does not contain enough information to answer,
say:

"I don't have enough information in the provided banking documents."

Context:
{context}

Question:
{question}
"""


# --------------------------------------------------
# 11. Generate answer
# --------------------------------------------------

response = llm.invoke(prompt)


# --------------------------------------------------
# 12. Display answer
# --------------------------------------------------

print("\n--- QUESTION ---")
print(question)

print("\n--- AI ANSWER ---")
print(response.content)


# --------------------------------------------------
# 13. Display sources
# --------------------------------------------------

print("\n--- SOURCES ---")

for document in relevant_results:

    source_name = document.metadata.get(
        "source_name",
        "Unknown source"
    )

    page_number = document.metadata.get("page_number")
    url = document.metadata.get("url")

    if page_number:
        print(
            f"{source_name} | "
            f"Page {page_number}"
        )

    elif url:
        print(
            f"{source_name} | "
            f"{url}"
        )

    else:
        print(source_name)