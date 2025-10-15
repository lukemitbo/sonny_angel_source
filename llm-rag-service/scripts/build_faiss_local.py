#!/usr/bin/env python3
"""
Script to build a FAISS index from local documents.
Usage: python build_faiss_local.py [--docs_dir DOCS_DIR] [--output_dir OUTPUT_DIR]
"""

import argparse
from pathlib import Path

from app.rag import LocalFaissVectorStoreManager, SimpleRetriever, read_texts_from_folder

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from local documents")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./docs",
        help="Directory containing documents to index (default: ./docs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/rag",
        help="Output directory for index files (default: ./artifacts/rag)"
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)

    if not docs_dir.exists():
        print(f"Error: Docs directory {docs_dir} does not exist")
        return

    # Initialize vector store
    vector_store = LocalFaissVectorStoreManager(str(output_dir))

    # Read documents
    print(f"Reading documents from {docs_dir}...")
    texts, metadatas = read_texts_from_folder(docs_dir)
    print(f"Found {len(texts)} documents")

    if not texts:
        print("No documents found to index")
        return

    # Build index
    print("Building FAISS index...")
    vector_store.add_texts(texts, metadatas)
    print(f"Index built and saved to {output_dir}")

    # Quick test
    print("\nTesting retrieval with a sample query...")
    retriever = SimpleRetriever(vector_store)
    results = retriever.retrieve("Who has Sonny Angel collaborated with?", k=2)
    print(f"Found {len(results)} results")
    for text, meta in results:
        print(f"\nSource: {meta.get('source_path')}")
        print(f"Text snippet: {text[:200]}...")

if __name__ == "__main__":
    main()
