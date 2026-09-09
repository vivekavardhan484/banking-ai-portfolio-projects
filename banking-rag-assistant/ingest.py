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


load_dotenv()


# --------------------------------------------------
# 1. Source configuration
# --------------------------------------------------

pdf_folder = Path("data/pdfs")

web_sources = [
    {
        "url": "https://www.fca.org.uk/consumers/fraudulent-payments",
        "source_name": "FCA Fraudulent Payments",
    }
]


# --------------------------------------------------
# 2. Load PDFs
# --------------------------------------------------

all_documents = []

pdf_files = list(pdf_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF file(s)")


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
            0
        )

        page.metadata["source_type"] = "pdf"
        page.metadata["source_name"] = pdf_file.stem
        page.metadata["file"] = pdf_file.name
        page.metadata["page_number"] = original_page + 1

    all_documents.extend(pages)


# --------------------------------------------------
# 3. Load and clean webpages
# --------------------------------------------------

for web_source in web_sources:

    url = web_source["url"]
    source_name = web_source["source_name"]

    print(f"\nLoading webpage: {url}")

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )


    # Remove unnecessary webpage elements
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


    # Prefer the main page content
    main_content = soup.find("main")

    if main_content is None:
        main_content = soup


    # Important:
    # separator=" " prevents words from different
    # HTML elements being joined together.
    clean_text = main_content.get_text(
        separator=" ",
        strip=True
    )


    # Remove excessive whitespace
    clean_text = " ".join(
        clean_text.split()
    )


    print(
        f"{source_name} | "
        f"Characters extracted: {len(clean_text)}"
    )


    document = Document(
        page_content=clean_text,
        metadata={
            "source_type": "web",
            "source_name": source_name,
            "url": url,
        }
    )

    all_documents.append(document)


# --------------------------------------------------
# 4. Check loaded documents
# --------------------------------------------------

if not all_documents:

    print("No usable documents were loaded.")
    raise SystemExit


print(
    f"\nLoaded {len(all_documents)} "
    f"document(s) total"
)


# --------------------------------------------------
# 5. Split into chunks
# --------------------------------------------------

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
    start=1
):

    chunk.metadata["chunk_number"] = i


# --------------------------------------------------
# 6. Delete old Chroma database
# --------------------------------------------------

db_path = Path("chroma_db")

if db_path.exists():

    shutil.rmtree(db_path)


# --------------------------------------------------
# 7. Create embeddings
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------------------------
# 8. Store everything in ChromaDB
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)


print(
    "\nEmbeddings stored successfully "
    "in ChromaDB!"
)