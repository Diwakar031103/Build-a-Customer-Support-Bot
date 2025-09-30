# Support Bot Agent (Document-trained Customer Support Bot)

## Objective
Build an AI-powered customer support bot in Python that:
- Reads and processes a provided document (TXT or PDF).
- Answers customer queries based on that document.
- Simulates feedback ("not helpful", "too vague") and iteratively improves responses.
- Logs decisions and actions for transparency.
- Handles out-of-scope queries gracefully.


## Features
- Load FAQ or company documents (TXT or PDF)
- Semantic search to find relevant sections (using `sentence-transformers`)
- Pre-trained NLP model for question answering (Hugging Face Transformers)
- Simulated feedback loop for response refinement
- Logging of queries, answers, and feedback
- Handles unknown/out-of-scope queries gracefully

## Installation

1. Clone this repository or download the code.
 * Install dependencies:
""" pip install transformers sentence-transformers PyPDF2 """



## Streamlit UI

<img width="661" height="85" alt="image" src="https://github.com/user-attachments/assets/8772a317-0df1-4841-8b0b-06d5861b78e6" />


## File Structure

<img width="745" height="176" alt="image" src="https://github.com/user-attachments/assets/19cd276e-0de1-4108-b783-5cd1ceb67561" />



## Usage
Console Version
* Bot reads faq.txt or any provided document.
* Sample queries will be executed.
* Responses and iterative improvements (based on feedback) will be printed in console.
* Logs saved in support_bot_log.txt.

## Streamlit Web App
<img width="754" height="45" alt="image" src="https://github.com/user-attachments/assets/a628099b-ba51-49ea-a477-63eddd856d25" />
* Upload your FAQ document (TXT or PDF)
* Type a query in the input box
* View initial and adjusted responses, along with feedback

## How It Works
1 Document Training
* Load the document (.txt or .pdf)
* Split into sections (paragraphs) for semantic retrieval
* Generate embeddings for sections (using sentence-transformers)

2 Query Handling
* Accept customer queries
* Retrieve the most relevant section
* Generate answer using QA model (distilbert-base-uncased-distilled-squad)

3 Feedback Loop
* Simulate feedback: "good", "too vague", "not helpful"
* Adjust responses: add context or rephrase
* Limit to 2 iterations per query

4 Logging
* All key actions logged in support_bot_log.txt (queries, selected sections, feedback, responses)

5 Fallback for Unknown Queries
* If query not covered in document:
* Bot responds: "I don’t have enough information to answer that."

## Sample Queries
"How do I reset my password?"

"What’s the refund policy?"

"How do I contact support?"

"How do I fly to the moon?" (out-of-scope example)

## Evaluation Criteria Covered

* Functionality: Trains on document, answers queries
* Adaptability: Adjusts answers based on feedback
* Code Quality: Modular, readable, well-commented
* Logging: Tracks key steps and decisions
* Robustness: Handles out-of-scope queries gracefully
* Documentation: Clear README with setup and usage


