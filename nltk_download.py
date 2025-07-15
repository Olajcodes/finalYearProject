from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Sample documents

doc1 = "This document is the Software Requirements Specification (SRS) for the insApption project. It details the functional and non-functional requirements of the system, as well as the goals of both the software and the organization. The document outlines the features the system should provide, such as language support, sub-app management, and user interface customization, and also specifies the different user interfaces that will be available. The document also provides an overview of the development, operation, and maintenance of the software, as well as the different roles of users and their associated requirements.The primary purpose of this document is to provide a detailed description of the insApption project, including its features, user interfaces, and development, operation, and maintenance processes. It also serves to outline the goals of both the software and the organization, as well as the different roles of users and their associated requirements. The document is intended to be used as a reference for stakeholders and developers of the insApption project, and to provide a comprehensive overview of the project."

doc2 = "The functional requirements of the system define its core capabilities and expected behaviors to ensure a seamless user experience. First, the system must support multilingual functionality, allowing all text elements in the user interface to be presented in the selected language. It is explicitly required to support both English and Dutch as available languages.Additionally, the system must enable the management of sub-applications within the main application. Specifically, sub-apps and their launchers should be added or deleted by a qualified programmer, ensuring controlled modifications within the software environment.Customization is another essential aspect of the system’s functionality. Users should be able to modify various visual elements within the user interface of both the main application and its sub-apps. These elements include colors, fonts, icons, buttons, and backgrounds. However, such modifications must be limited to internal resources, excluding any externally sourced content.Overall, these functional requirements ensure that the system remains adaptable, user-friendly, and manageable, while providing essential customization features to enhance usability and maintainability"

# Vectorize the documents

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# print(tfidf_matrix)

# Calculate cosine similarity

cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

print(f"Cosine Similarity: {cosine_sim[0][1]}")

best_model = cosine_sim.mean(axis=0).idxmax()

