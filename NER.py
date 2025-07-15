from IPython.display import display
import os
import docx
import pandas as pd
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline

# Initialize the Hugging Face NER pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_from_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_excel(file_path: str) -> str:
    df = pd.read_excel(file_path, header=None)
    return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1))

def extract_named_entities(text: str) -> dict:
    results = ner_pipeline(text)
    entities = {}
    for ent in results:
        label = ent["entity_group"]
        if label not in entities:
            entities[label] = set()
        entities[label].add(ent["word"])
    return {label: list(vals) for label, vals in entities.items()}

def summarize_text(text: str, num_sentences: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def write_summary_output(output_path: str, entities: dict, summary: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Named Entities\n")
        for label, items in entities.items():
            f.write(f"\n## {label}\n")
            for item in items:
                f.write(f"- {item}\n")
        f.write("\n# Summary\n\n")
        f.write(summary)

def main():
    input_path = r"C:\Users\Welcome Sir\Desktop\Final_year Project\pythonCode\datasets\IBM DOORS\Original.docx"
    output_path = "./srs_summary.txt"

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    text = extract_text_from_file(input_path)
    entities = extract_named_entities(text)
    summary = summarize_text(text)
    write_summary_output(output_path, entities, summary)
    print(f"Summary saved to: {output_path}")

if __name__ == "__main__":
    main()
