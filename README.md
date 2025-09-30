# Support Bot Agent (Document-trained Customer Support Bot)

This project provides a simple customer support bot that:

- Loads a provided document (`.txt` or `.pdf`), 
- Uses sentence-transformers for semantic retrieval,
- Uses a Hugging Face QA pipeline to answer queries based on the retrieved context,
- Simulates feedback and iteratively improves the response (up to 2 iterations),
- Logs actions to `support_bot_log.txt`.

## Files
- `support_bot_agent.py` — The main script containing `SupportBotAgent` class and a demo `__main__` block.
- `faq.txt` — Sample FAQ used by the demo (auto-created if missing).
- `requirements.txt` — Python package requirements.
- `support_bot_log.txt` — Created at runtime (contains logs).

## Setup
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate    # Windows (PowerShell)
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the demo
```bash
python support_bot_agent.py
```
This will create `faq.txt` (if missing), load models, and run sample queries. Output will be printed to console and a log will be written to `support_bot_log.txt`.

## Notes / Next steps
- For large documents, consider splitting into smaller chunks (by paragraph or fixed-size windows).
- You can change the QA or embedder model names when creating `SupportBotAgent`.
- To push to GitHub: initialize a repo, add files and push.
```bash
git init
git add support_bot_agent.py requirements.txt README.md faq.txt
git commit -m "Initial commit - support bot project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
