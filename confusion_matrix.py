# Import necessary libraries
import os
import fitz  # PyMuPDF for reading PDF files
import docx  # For reading Word documents
import pandas as pd  # For reading Excel files and data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical features
from sklearn.metrics.pairwise import cosine_similarity  # For computing cosine similarity between texts

# Main function to extract text based on file type
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()  # Get file extension
    if ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".txt":
        return extract_text_txt(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_excel(file_path)
    else:
        raise ValueError("Unsupported file type")  # Raise error for unsupported file types

# Extract text from a PDF file
def extract_text_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()  # Concatenate text from each page
    return text

# Extract text from a Word (.docx) file
def extract_text_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)  # Join all paragraphs

# Extract text from a plain text (.txt) file
def extract_text_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback for alternative encoding if UTF-8 fails
        with open(file_path, "r", encoding="cp1252") as f:
            return f.read()

# Extract text from an Excel file by reading all cells row-wise
def extract_text_excel(file_path: str) -> str:
    df = pd.read_excel(file_path, header=None)
    return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1))  # Convert to string and join row-wise

# Compute cosine similarity matrix from a list of text documents
def compute_cosine_similarity(texts: list[str]) -> list[list[float]]:
    vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform(texts)  # Transform texts to TF-IDF vectors
    return cosine_similarity(tfidf_matrix)  # Compute pairwise cosine similarity

# Plot and save the cosine similarity matrix as a heatmap
def plot_confusion_matrix(sim_matrix, labels, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap="Blues", fmt=".2f")
    plt.title("Cosine Similarity Matrix (IBM DOORS)")
    plt.tight_layout()
    plt.savefig(output_file)  # Save figure as PNG
    print(f"Matrix saved to: {output_file}")

# Main execution function
def main():
    # Dictionary of summary labels and their corresponding file paths
    summary_paths = {
        "Copilot": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\Copilot.txt",
        "ChatGPT": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\ChatGPT.txt",
        "Gemini": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\Gemini.txt",
        "LDA": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\LDA_summary.pdf",
        "NER": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\IBM_summary.txt",
        "Expert": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\Domain-expert.txt",
        "IBM_Reference": r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\Original.docx",
    }

    texts = []  # Store extracted texts
    labels = []  # Store corresponding labels

    # Extract text from each file and store with its label
    for label, path in summary_paths.items():
        labels.append(label)
        texts.append(extract_text(path))

    # Compute similarity matrix
    sim_matrix = compute_cosine_similarity(texts)
    
     # Convert to DataFrame for table format
    sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    # Print similarity scores in table format
    print("\nCosine Similarity Table:\n")
    print(sim_df)

    # Plot and save the matrix
    plot_confusion_matrix(sim_matrix, labels, "2IBM_cosine_similarity_matrix.png")

# Run the script
if __name__ == "__main__":
    main()
