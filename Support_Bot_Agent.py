
--------------- important libraries ---------------------

""" For logging events, errors, and process info into a log file """
    import logging
""" For generating random feedback (like 'good', 'too vague', etc.) """
    import random
""" For file handling (checking/creating FAQ file) """
    import os
""" For type hints (e.g., list of queries) """
    from typing import List
""" For loading the Question Answering model """
    from transformers import pipeline
""" For embeddings and similarity search """
    from sentence_transformers import SentenceTransformer, util
""" For reading and extracting text from PDF files """
    import PyPDF2

-----------------logging Configuration-----------------------------------
""" Saves all activities (queries, errors, answers) into a log file """

    logging.basicConfig(
        filename='support_bot_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

---------------- Support Bot Class ---------------

    class SupportBotAgent:
        def __init__(self, document_path: str, qa_model_name: str = "distilbert-base-uncased-distilled-squad", embed_model_name: str = "all-MiniLM-L6-v2"):
         
            """Initialize QA pipeline and embedder, load document and compute embeddings."""
            """ Initialize models """
            
            self.qa_model = pipeline("question-answering", model=qa_model_name)
            self.embedder = SentenceTransformer(embed_model_name)

            """ Load and preprocess document """
            self.document_text = self.load_document(document_path)
            
            """ Split document into sections (by paragraphs) """
            self.sections = [s.strip() for s in self.document_text.split('\n\n') if s.strip()]
            if not self.sections:
                
               """ If no paragraphs, fallback to splitting by lines """
                self.sections = [s.strip() for s in self.document_text.split('\n') if s.strip()]

            """ Create embeddings for each section """
            try:
                self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
            except Exception as e:
              """ If GPU or tensor issue occurs, fallback to plain numpy embeddings """
                logging.warning(f"Embedding encode failed with: {e}. Retrying without convert_to_tensor.")
                embeddings = self.embedder.encode(self.sections, convert_to_tensor=False)
                self.section_embeddings = embeddings

            logging.info(f"Loaded document: {document_path} | Sections: {len(self.sections)}")

        def load_document(self, path: str) -> str:
            """ Loads text from a document (supports .txt or .pdf files). """
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


--------------------- Find Relevant Section ----------------
""" Finds the most relevant section of the document for a given query
        using semantic similarity (cosine similarity of embeddings). """

        def find_relevant_section(self, query: str) -> str:
            """Return the most semantically relevant section for the query."""
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            # util.cos_sim works with tensors or numpy arrays
            similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
            best_idx = int(similarities.argmax())
            best_section = self.sections[best_idx]
            logging.info(f"Query: '{query}' | Selected section index: {best_idx}")
            return best_section

 ------------------- Answer Query ----------------
        def answer_query(self, query: str) -> str:
            """Generate an answer for the query using a retrieved context."""
            """ Retrieve the best section """
            try:
                context = self.find_relevant_section(query)
            except Exception as e:
                logging.exception(f"Failed to find relevant section: {e}")
                return "I don't have enough information to answer that."

            if not context or len(context.strip()) < 10:
                return "I don't have enough information to answer that."

            """ Use the QA model """
            try:
                result = self.qa_model(question=query, context=context)
                
                """ result can be a dict {answer, score, start, end} """
                answer = result.get('answer') if isinstance(result, dict) else str(result)
            except Exception as e:
                logging.exception(f"QA model failed: {e}")
                return "I encountered an error generating an answer."

            logging.info(f"Answer generated for query '{query}': {answer[:120]}...")
            return answer
            
 ---------------- Feedback Simulation ------------------

        def get_feedback(self, response: str) -> str:
            """ Simulate feedback. Very short responses considered 'too vague'."""
            if not response or len(response.split()) < 3:
                feedback = 'too vague'
            else:
                feedback = random.choice(['not helpful', 'too vague', 'good'])
            logging.info(f"Simulated feedback: {feedback}")
            return feedback

---------------- Adjust Response -------------------

        def adjust_response(self, query: str, response: str, feedback: str) -> str:
            """ Adjust the response based on feedback. Limited to rephrasing or adding context. """
            if feedback == 'too vague':
                context = self.find_relevant_section(query)
                # Provide a bit more context from the source
                snippet = context[:200].strip()
                return f"{response} Additional context: {snippet}..."
            elif feedback == 'not helpful':
                
                """ Ask the QA model to answer again with a prompt to be more specific"""
                follow_up_query = query + " Please be more specific and provide steps or examples."
                return self.answer_query(follow_up_query)
            return response
            
------------------- Run Queries --------------------
        def run(self, queries: List[str], max_iterations: int = 2):
            """ Process a list of queries, simulate feedback, and iterate to improve."""
            for query in queries:
                logging.info(f"Processing query: {query}")
                response = self.answer_query(query)
                print(f"Initial Response to '{query}': {response}")

                """ Feedback loop """
                for iteration in range(max_iterations):
                    feedback = self.get_feedback(response)
                    if feedback == 'good':
                        print(f"Final Response to '{query}': {response}")
                        break
                    response = self.adjust_response(query, response, feedback)
                    print(f"Updated Response to '{query}' (iter {iteration+1}): {response}")


---------------- Main Execution ----------------

    if __name__ == '__main__':
        sample_faq = 'faq.txt'
        
        """ Create a sample FAQ file if it does not exist """
        if not os.path.exists(sample_faq):
            with open(sample_faq, 'w', encoding='utf-8') as f:
                f.write("""Resetting Your Password
To reset your password, go to the login page and click "Forgot Password".
Enter your registered email and follow the password reset link sent to you.
Refund Policy
We offer refunds within 30 days of purchase. Please contact support@example.com with your order number to start the refund process.
Contacting Support
You can contact our support team via email at support@example.com or call 1-800-555-1234 during business hours (9 AM - 5 PM EST).
Shipping Information
Orders are processed within 2 business days. Shipping usually takes 5-7 business days. You will receive a tracking number by email once your order has shipped.
Account Deletion
If you want to permanently delete your account, please send a request to support@example.com. 
Your account and associated data will be removed within 14 days.\n""")
                print(f"Sample FAQ created at {sample_faq}")


    """ Initialize bot and run sample queries """"
        bot = SupportBotAgent(sample_faq)
        sample_queries = [
            "How do I reset my password?",
            "What\u2019s the refund policy?",
            "How do I contact support?",
            "How do I fly to the moon?"
        ]
        bot.run(sample_queries)
