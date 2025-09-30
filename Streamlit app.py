------------------- Install Dependencies -------------------
""" Run this once in terminal before starting """
""" pip install streamlit transformers sentence-transformers PyPDF2 """

import streamlit as st
import logging
import random
import os
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

---------------- Logging Setup ----------------
logging.basicConfig(
    filename="support_bot_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

---------------- Bot Class ----------------
class SupportBotAgent:
    def __init__(self, document_path):
        # Load QA and embedding models
        self.qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load document
        self.document_text = self.load_document(document_path)

        # Split document into sections for retrieval
        self.sections = [s.strip() for s in self.document_text.split("\n\n") if s.strip()]
        self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
        logging.info(f"Loaded document: {document_path}")

    def load_document(self, path):
        """Load .txt or .pdf file into plain text"""
        text = ""
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
        elif path.endswith(".pdf"):
            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            raise ValueError("Unsupported file format. Use .txt or .pdf")
        return text

    def find_relevant_section(self, query):
        """Find the most relevant section based on similarity"""
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_section = self.sections[best_idx]
        logging.info(f"Query: '{query}' | Section: {best_section[:80]}...")
        return best_section

    def answer_query(self, query):
        """Answer user query using the best-matching section"""
        context = self.find_relevant_section(query)
        if not context.strip():
            return "Sorry, I don’t have enough information to answer that. Please contact support@example.com."
        result = self.qa_model(question=query, context=context)
        return result["answer"]

    def get_feedback(self, response):
        """Simulate feedback"""
        if len(response.split()) < 3:
            feedback = "too vague"
        else:
            feedback = random.choice(["not helpful", "too vague", "good"])
        logging.info(f"Feedback: {feedback}")
        return feedback

    def adjust_response(self, query, response, feedback):
        """Adjust response based on feedback"""
        if feedback == "too vague":
            context = self.find_relevant_section(query)
            return f"{response} (More info: {context[:120]}...)"
        elif feedback == "not helpful":
            return self.answer_query(query + " Please provide more details.")
        return response

    def run(self, query):
        """Run query + feedback loop"""
        response = self.answer_query(query)
        feedbacks = []
        responses = [f"Initial Response: {response}"]

        for _ in range(2):  # max 2 iterations
            feedback = self.get_feedback(response)
            feedbacks.append(feedback)
            if feedback == "good":
                responses.append(f"Final Response: {response}")
                break
            response = self.adjust_response(query, response, feedback)
            responses.append(f"Updated Response: {response}")

        return responses, feedbacks

 ---------------- Streamlit UI ----------------
st.title("🤖 Customer Support Bot")

""" Upload FAQ document """
uploaded_file = st.file_uploader("Upload your FAQ document (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file:
    # Save uploaded file locally
    file_ext = uploaded_file.name.split(".")[-1]
    file_path = "uploaded_doc." + file_ext
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Document uploaded successfully!")

    """ Initialize bot """
    bot = SupportBotAgent(file_path)

    """ User query input """
    query = st.text_input("Ask your question:")

    if query:
        responses, feedbacks = bot.run(query)
        for i, r in enumerate(responses):
            st.write(f"**{r}**")
            if i < len(feedbacks):
                st.caption(f"Feedback: {feedbacks[i]}")
