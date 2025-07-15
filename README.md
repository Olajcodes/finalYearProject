# 📄 Performance Evaluation of Abstraction Agents on Software Requirements Specifications (SRS) Summarization 

## 🧠 Abstract
The complexity and verbosity of Software Requirements Specification (SRS) documents pose a major challenge to stakeholders. While essential for guiding system design and implementation, these documents are often hard to digest quickly, resulting in miscommunication and project delays.

This study evaluates six summarization agents — ChatGPT, Copilot, Gemini, LDA, BERT, and a domain expert — on two SRS datasets (IBM DOORS and Mendeley). Using cosine similarity, verbosity metrics, and visualization tools, the project assesses the balance between semantic retention and output conciseness. Results show that generative AI models, especially Gemini and ChatGPT, outperform traditional methods and offer scalable summarization solutions.

The study proposes a hybrid summarization approach: leveraging generative AI for initial summarization followed by expert refinement for domain accuracy.

## 📌 Features
- Evaluation of 6 summarization agents on real-world SRS datasets
- Use of cosine similarity for semantic comparison
- Output verbosity analysis (word count statistics)
- Visual representation using similarity matrices and bar charts
- Python-based evaluation pipeline with modular design

## 🧰 Tech Stack
- Python 3.x
- Hugging Face Transformers
- OpenAI API (for ChatGPT)
- Gemini API (Google)
- Microsoft Copilot API
- matplotlib / seaborn
- NLTK / spaCy
- Pandas, NumPy

## 📂 Project Structure

finalYearCode/
│
├── datasets/
│   ├── IBM_DOORS/
│   │   ├── ChatGPT.txt
│   │   ├── Copilot.txt
│   │   ├── Gemini.txt
│   │   ├── Domain-expert.txt
│   │   ├── LDA_summary.pdf
│   │   ├── Original.docx
│   │   └── IBM_summary.txt
│   │
│   ├── Mendeley/
│   │   ├── ChatGPT.txt
│   │   ├── Copilot.txt
│   │   ├── Gemini.txt
│   │   ├── Domain-expert.txt
│   │   ├── LDA_summary.pdf
│   │   ├── Original.xlsx
│   │   └── Mendeley_summary.txt
│
├── scripts/
│   ├── confusion_matrix_ibm.py           # From confusion_matrix.py
│   ├── confusion_matrix_mendeley.py      # From mendeley_confusion.py
│   ├── cosine_similarity_single.py       # From cosineSimilarity.py
│   ├── NER.py                            # From NER.py
│
├── results/
│   ├── IBM_cosine_similarity_matrix.png
│   ├── Mendeley_cosine_similarity_matrix.png
│
├── README.md
├── requirements.txt
└── .gitignore

## 👤 Author
Abioye Olajide Abdullateef
Computer Science Graduate
Federal University of Agriculture, Abeokuta

## ## Supervisor
Mr. T.O OLALEYE
Computer Science Lecturer
Federal University of Agriculture, Abeokuta
