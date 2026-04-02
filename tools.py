from langchain_core.tools import tool
from datetime import datetime

vectorstore = None


@tool
def get_time() -> str:
    """Get current system time"""
    return datetime.now().strftime("%H:%M")


@tool
def search_pdf(query: str) -> str:
    """Search information from stored PDF"""

    global vectorstore

    if vectorstore is None:
        return "Vectorstore not initialized."

    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        return "No relevant information found."

    return "\n\n".join([doc.page_content for doc in docs])