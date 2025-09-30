    import logging
    import random
    import os
    from typing import List
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer, util
    import PyPDF2

    # Configure logging
    logging.basicConfig(
        filename='support_bot_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    class SupportBotAgent:
        def __init__(self, document_path: str, qa_model_name: str = "distilbert-base-uncased-distilled-squad", embed_model_name: str = "all-MiniLM-L6-v2"):
            """Initialize QA pipeline and embedder, load document and compute embeddings."""
            # Initialize models
            self.qa_model = pipeline("question-answering", model=qa_model_name)
            self.embedder = SentenceTransformer(embed_model_name)

            # Load and preprocess document
            self.document_text = self.load_document(document_path)
            # Split document into sections (by double-newline paragraphs) and filter empties
            self.sections = [s.strip() for s in self.document_text.split('\n\n') if s.strip()]
            if not self.sections:
                # fallback: split by lines
                self.sections = [s.strip() for s in self.document_text.split('\n') if s.strip()]

            # Compute embeddings for retrieval
            try:
                self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
            except Exception as e:
                # If GPU issues or similar, fallback to CPU numpy embeddings
                logging.warning(f"Embedding encode failed with: {e}. Retrying without convert_to_tensor.")
                embeddings = self.embedder.encode(self.sections, convert_to_tensor=False)
                # convert to tensor on-the-fly when needed
                self.section_embeddings = embeddings

            logging.info(f"Loaded document: {document_path} | Sections: {len(self.sections)}")

        def load_document(self, path: str) -> str:
            """Load text from .txt or .pdf file."""
            text = ""
            if path.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif path.lower().endswith('.pdf'):
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                    text = '\n'.join(pages)
            else:
                raise ValueError('Unsupported file format. Use .txt or .pdf')
            return text

        def find_relevant_section(self, query: str) -> str:
            """Return the most semantically relevant section for the query."""
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            # util.cos_sim works with tensors or numpy arrays
            similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
            best_idx = int(similarities.argmax())
            best_section = self.sections[best_idx]
            logging.info(f"Query: '{query}' | Selected section index: {best_idx}")
            return best_section

        def answer_query(self, query: str) -> str:
            """Generate an answer for the query using a retrieved context."""
            # Retrieve the best section
            try:
                context = self.find_relevant_section(query)
            except Exception as e:
                logging.exception(f"Failed to find relevant section: {e}")
                return "I don't have enough information to answer that."

            if not context or len(context.strip()) < 10:
                return "I don't have enough information to answer that."

            # Use the QA model
            try:
                result = self.qa_model(question=query, context=context)
                # result can be a dict {answer, score, start, end}
                answer = result.get('answer') if isinstance(result, dict) else str(result)
            except Exception as e:
                logging.exception(f"QA model failed: {e}")
                return "I encountered an error generating an answer."

            logging.info(f"Answer generated for query '{query}': {answer[:120]}...")
            return answer

        def get_feedback(self, response: str) -> str:
            """Simulate feedback. Very short responses considered 'too vague'."""
            if not response or len(response.split()) < 3:
                feedback = 'too vague'
            else:
                feedback = random.choice(['not helpful', 'too vague', 'good'])
            logging.info(f"Simulated feedback: {feedback}")
            return feedback

        def adjust_response(self, query: str, response: str, feedback: str) -> str:
            """Adjust the response based on feedback. Limited to rephrasing or adding context."""
            if feedback == 'too vague':
                context = self.find_relevant_section(query)
                # Provide a bit more context from the source
                snippet = context[:200].strip()
                return f"{response} Additional context: {snippet}..."
            elif feedback == 'not helpful':
                # Ask the QA model to answer again with a prompt to be more specific
                follow_up_query = query + " Please be more specific and provide steps or examples."
                return self.answer_query(follow_up_query)
            return response

        def run(self, queries: List[str], max_iterations: int = 2):
            """Process a list of queries, simulate feedback, and iterate to improve."""
            for query in queries:
                logging.info(f"Processing query: {query}")
                response = self.answer_query(query)
                print(f"Initial Response to '{query}': {response}")

                # Feedback loop
                for iteration in range(max_iterations):
                    feedback = self.get_feedback(response)
                    if feedback == 'good':
                        print(f"Final Response to '{query}': {response}")
                        break
                    response = self.adjust_response(query, response, feedback)
                    print(f"Updated Response to '{query}' (iter {iteration+1}): {response}")

    if __name__ == '__main__':
        # Demo usage
        sample_faq = 'faq.txt'
        # Create sample faq if missing (safe-guard)
        if not os.path.exists(sample_faq):
            with open(sample_faq, 'w', encoding='utf-8') as f:
                f.write("""Resetting Your Password
To reset your password, go to the login page and click \"Forgot Password.\"\nEnter your registered email and follow the password reset link sent to you.\n\nRefund Policy\nWe offer refunds within 30 days of purchase. Please contact support@example.com with your order number to start the refund process.\n\nContacting Support\nYou can contact our support team via email at support@example.com or call 1-800-555-1234 during business hours (9 AM - 5 PM EST).\n\nShipping Information\nOrders are processed within 2 business days. Shipping usually takes 5-7 business days. You will receive a tracking number by email once your order has shipped.\n\nAccount Deletion\nIf you want to permanently delete your account, please send a request to support@example.com. Your account and associated data will be removed within 14 days.\n""")
                print(f"Sample FAQ created at {sample_faq}")

        bot = SupportBotAgent(sample_faq)
        sample_queries = [
            "How do I reset my password?",
            "What\u2019s the refund policy?",
            "How do I contact support?",
            "How do I fly to the moon?"
        ]
        bot.run(sample_queries)
