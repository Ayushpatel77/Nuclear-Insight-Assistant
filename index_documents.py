"""
Run this ONCE before starting the app.
It indexes all PDFs/TXTs in data/docs/ into the FAISS index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.document_processor import ingest_documents, get_indexed_documents

DOCS_DIR = Path(__file__).parent / "data" / "docs"

def main():
    files = list(DOCS_DIR.glob("*.pdf")) + \
            list(DOCS_DIR.glob("*.txt")) + \
            list(DOCS_DIR.glob("*.md"))

    if not files:
        print(f"❌ No documents found in {DOCS_DIR}")
        print("   Put your PDF/TXT files in data/docs/ and run this again.")
        sys.exit(1)

    print(f" Found {len(files)} document(s):")
    for f in files:
        print(f"   - {f.name}")
    print()

    result = ingest_documents([str(f) for f in files], status_cb=print)

    if result["status"] == "ok":
        print(f"\n Done! Indexed {result['total_chunks']} chunks from {len(result['documents'])} document(s).")
        print("   Now run: streamlit run app.py")
    else:
        print(f"\n Error: {result['message']}")

if __name__ == "__main__":
    main()
