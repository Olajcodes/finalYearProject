import os
import fitz  # PyMuPDF for PDFs
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    """Reads content from a PDF or DOCX file."""
    if file_path.lower().endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

def read_pdf(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text

def read_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def process_documents(doc1, doc2):
    """Computes cosine similarity between two text documents."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    print(f"Cosine Similarity: {cosine_sim[0][1]}")

def main():
    """Accepts input from users for file paths or raw text."""
    input1 = input("Enter the reference file path: ")
    input2 = input("Enter the generated summary text or file path: ")
    
    reference_file = read_file(input1) if os.path.isfile(input1) else input1
    generated_summary = read_file(input2) if os.path.isfile(input2) else input2
    
    process_documents(reference_file, generated_summary)

if __name__ == "__main__":
    main()