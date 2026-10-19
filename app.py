from rag.pdf_loader import load_pdfs

documnets = load_pdfs("data")

for doc in documnets:
    print("FILE:", doc["filename"])
    print(doc["text"][:500])
    print("-" * 50)
