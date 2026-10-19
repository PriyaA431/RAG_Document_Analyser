
from rag.pdf_loader import load_pdfs
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.chunker import chunk_text
from rag.llm import LocalLLM

def main():
    # 1. Load PDFs
    documents = load_pdfs("data")

    if not documents:
        print("No PDFs found in the data folder.")
        return

    print("PDFs loaded:", len(documents))

    # 2. Split PDF text into chunks
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f'{doc["filename"]}_{index}',
                "text": chunk,
                "source": doc["filename"],
                "chunk_index": index
            })

    print("Total chunks:", len(all_chunks))

    if not all_chunks:
        print("No text chunks created. Check PDF text extraction.")
        return

    # 3. Generate embeddings
    embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in all_chunks]
    print("Texts for embedding:", len(texts))

    embeddings = embedding_model.create_embeddings(texts)

    print("Embeddings generated:", len(embeddings))

    if len(embeddings) != len(all_chunks):
        print("Error: Chunk and embedding counts do not match.")
        return

    # 4. Store chunks and embeddings
    vector_store = VectorStore()
    vector_store.add_chunks(all_chunks, embeddings)

    print("Total chunks in database:", vector_store.collection.count())


    # 5. Retrieve relevant chunks
    
    llm = LocalLLM()

    print("\nPDF Assistant is ready!")
    print("Ask questions about your PDFs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        if not question:
            continue

        query_embedding = embedding_model.create_embeddings([question])[0]

        results = vector_store.search(query_embedding, top_k=5)

        retrieved_documents = results["documents"][0]
        metadata = results["metadatas"][0]

        if not retrieved_documents:
            print("Assistant: I couldn't find relevant information in your PDFs.")
            continue

        context = "\n\n".join(retrieved_documents)

        answer = llm.generate_answer(question, context)

        print(f"\nAssistant: {answer}\n")

        sources = sorted(set(item["source"] for item in metadata))
        print("Sources:", ", ".join(sources), "\n")




if __name__ == "__main__":
    main()
