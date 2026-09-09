from pathlib import Path
import shutil
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ==================================================
# Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_FOLDER = BASE_DIR / "data" / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv()


WEB_SOURCES = [
    {
        "url": "https://www.fca.org.uk/consumers/fraudulent-payments",
        "source_name": "FCA Fraudulent Payments",
    }
]


# ==================================================
# Load documents
# ==================================================

all_documents = []

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF file(s)")


# ==================================================
# PDF ingestion
# ==================================================

for pdf_file in pdf_files:

    print(f"\nLoading PDF: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pages = loader.load()

    print(f"Pages extracted: {len(pages)}")


    for i, page in enumerate(pages, start=1):

        print(
            f"{pdf_file.name} | "
            f"Page {i} | "
            f"Characters extracted: {len(page.page_content)}"
        )


    for page in pages:

        original_page = page.metadata.get(
            "page",
            0,
        )

        # Friendly source name for portfolio display
        if pdf_file.name.lower() == "tr17-1.pdf":

            source_name = (
                "FCA TR17/1 – Customer Understanding "
                "in Retail Banking"
            )

        else:

            source_name = pdf_file.stem


        page.metadata["source_type"] = "pdf"

        page.metadata["source_name"] = source_name

        page.metadata["file"] = pdf_file.name

        page.metadata["page_number"] = (
            original_page + 1
        )


    all_documents.extend(pages)


# ==================================================
# Web ingestion
# ==================================================

for web_source in WEB_SOURCES:

    url = web_source["url"]

    source_name = web_source["source_name"]


    print(
        f"\nLoading webpage: {url}"
    )


    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()


    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )


    # Remove page elements that are not useful
    # for the knowledge base
    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
        ]
    ):

        element.decompose()


    main_content = soup.find("main")

    if main_content is None:
        main_content = soup


    clean_text = main_content.get_text(
        separator=" ",
        strip=True,
    )

    clean_text = " ".join(
        clean_text.split()
    )


    print(
        f"{source_name} | "
        f"Characters extracted: "
        f"{len(clean_text)}"
    )


    document = Document(
        page_content=clean_text,
        metadata={
            "source_type": "web",
            "source_name": source_name,
            "url": url,
        },
    )


    all_documents.append(document)


# ==================================================
# Validation
# ==================================================

if not all_documents:

    print(
        "No usable documents were loaded."
    )

    raise SystemExit


print(
    f"\nLoaded "
    f"{len(all_documents)} document(s) total"
)


# ==================================================
# Chunking
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
)


chunks = text_splitter.split_documents(
    all_documents
)


print(
    f"Created {len(chunks)} chunks"
)


for i, chunk in enumerate(
    chunks,
    start=1,
):

    chunk.metadata["chunk_number"] = i


# ==================================================
# Rebuild Chroma database
# ==================================================

if CHROMA_DIR.exists():

    print(
        "\nRemoving old Chroma database..."
    )

    shutil.rmtree(CHROMA_DIR)


print(
    "Creating new embeddings..."
)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
)


print(
    "\nEmbeddings stored successfully "
    "in ChromaDB!"
)

print(
    f"Database location: {CHROMA_DIR}"
)