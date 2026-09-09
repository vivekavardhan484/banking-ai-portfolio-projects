from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# --------------------------------------------------
# 1. Connect to ChromaDB
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)


# --------------------------------------------------
# 2. Evaluation questions
# --------------------------------------------------

test_cases = [
    {
        "question": "What should I do if I sent money to a scammer?",
        "expected_source": "FCA Fraudulent Payments",
        "should_retrieve": True,
    },
    {
        "question": "What should I do if I notice an unauthorised payment?",
        "expected_source": "FCA Fraudulent Payments",
        "should_retrieve": True,
    },
    {
        "question": "Can victims of APP fraud receive reimbursement?",
        "expected_source": "FCA Fraudulent Payments",
        "should_retrieve": True,
    },
    {
        "question": "What problems can customers have understanding retail banking products?",
        "expected_source": "tr17-1",
        "should_retrieve": True,
    },
    {
        "question": "Why is customer understanding important for retail banks?",
        "expected_source": "tr17-1",
        "should_retrieve": True,
    },
    {
        "question": "What is the current Bitcoin price?",
        "expected_source": None,
        "should_retrieve": False,
    },
    {
        "question": "What will the weather be tomorrow?",
        "expected_source": None,
        "should_retrieve": False,
    },
    {
        "question": "What mortgage interest rate does this bank offer?",
        "expected_source": None,
        "should_retrieve": False,
    },
]


# --------------------------------------------------
# 3. Retrieval settings
# --------------------------------------------------

TOP_K = 4

# Provisional threshold.
# We will evaluate whether 0.3 is actually suitable.
THRESHOLD = 0.3


# --------------------------------------------------
# 4. Run evaluation
# --------------------------------------------------

passed = 0

print("\n========================================")
print("       RAG RETRIEVAL EVALUATION")
print("========================================")


for number, test in enumerate(test_cases, start=1):

    question = test["question"]

    results = (
        vector_store.similarity_search_with_relevance_scores(
            question,
            k=TOP_K
        )
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= THRESHOLD
    ]

    retrieved_sources = {
        document.metadata.get(
            "source_name",
            "Unknown source"
        )
        for document, score in relevant_results
    }


    # ----------------------------------------------
    # Determine PASS / FAIL
    # ----------------------------------------------

    if test["should_retrieve"]:

        success = (
            len(relevant_results) > 0
            and test["expected_source"]
            in retrieved_sources
        )

    else:

        success = (
            len(relevant_results) == 0
        )


    if success:
        passed += 1
        status = "PASS"
    else:
        status = "FAIL"


    # ----------------------------------------------
    # Print result
    # ----------------------------------------------

    print(f"\nTest {number}: {status}")

    print(f"Question:")
    print(question)

    print(
        f"Expected source: "
        f"{test['expected_source']}"
    )

    print(
        f"Should retrieve: "
        f"{test['should_retrieve']}"
    )


    print("\nTop retrieval results:")

    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        source_name = document.metadata.get(
            "source_name",
            "Unknown source"
        )

        page_number = document.metadata.get(
            "page_number"
        )

        if page_number:
            location = f"Page {page_number}"
        else:
            location = "Webpage"

        threshold_status = (
            "KEPT"
            if score >= THRESHOLD
            else "REJECTED"
        )

        print(
            f"  {i}. "
            f"{score:.3f} | "
            f"{source_name} | "
            f"{location} | "
            f"{threshold_status}"
        )


# --------------------------------------------------
# 5. Final score
# --------------------------------------------------

total = len(test_cases)

accuracy = (
    passed / total
) * 100


print("\n========================================")
print("          EVALUATION SUMMARY")
print("========================================")

print(f"Passed: {passed}/{total}")
print(f"Retrieval accuracy: {accuracy:.1f}%")

print("========================================\n")