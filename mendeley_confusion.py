import os
import fitz  # PyMuPDF
import docx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Determines which file extraction function to use based on file extension
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".txt":
        return extract_text_txt(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_excel(file_path)
    else:
        raise ValueError("Unsupported file type")


# Extract text from a PDF using PyMuPDF
def extract_text_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# Extract text from a .docx file
def extract_text_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


# Extract text from a plain .txt file
def extract_text_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp1252") as f:
            return f.read()


# Extract text from Excel by joining all cells as strings
def extract_text_excel(file_path: str) -> str:
    df = pd.read_excel(file_path, header=None)
    return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1))


# Compute cosine similarity between a list of text documents
def compute_cosine_similarity(texts: list[str]) -> list[list[float]]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)


# Plot the cosine similarity as a heatmap matrix
def plot_confusion_matrix(sim_matrix, labels, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap="Reds", fmt=".2f")
    plt.title("Cosine Similarity Matrix (Mendeley Dataset)")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Matrix saved to: {output_file}")


def main():
    # Define paths to all summary files
    summary_paths = {
        "Copilot": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\Copilot.txt",
        "ChatGPT": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\ChatGPT.txt",
        "Gemini": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\Gemini.txt",
        "LDA": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\LDA_summary.pdf",
        "NER": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\Mendeley_summary.txt",
        "Expert": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\Domain-expert.txt",
        "Mendeley_Reference": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\Mendeley Data\Original.xlsx",
    }

    texts = []
    labels = []

    # Extract text from all provided paths
    for label, path in summary_paths.items():
        labels.append(label)
        texts.append(extract_text(path))

    # Compute cosine similarity matrix
    sim_matrix = compute_cosine_similarity(texts)

    # Convert to DataFrame for table format
    sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    # Print similarity scores in table format
    print("\nCosine Similarity Table:\n")
    print(sim_df)


    # Plot and save the heatmap
    plot_confusion_matrix(sim_matrix, labels, "New_Mendeley Dataset_cosine_similarity_matrix.png")


if __name__ == "__main__":
    main()
