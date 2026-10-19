from pypdf import PdfReader
from pathlib import Path

def load_pdfs(data_folder: str):
    """ Extract text from PDF  """
    documents = []

    for pdf_path in Path(data_folder).glob("*.pdf"):

        reader = PdfReader(pdf_path)

        text = "" 

        for page in reader.pages:
            page_text = page.extract_text()


            if page_text:
                text += page_text +  "\n"

        documents.append({
            "filename" : pdf_path.name,
            "text" : text
        })

    return documents