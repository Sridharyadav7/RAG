# 🔍 RAG API — Retrieval-Augmented Generation with FastAPI

## 🚀 Quickstart

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Sridharyadav7/RAG.git
cd RAG
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Create a .env file and add your API keys with reference to env.txt

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the load_pdfs.py to index documents present in pdfs folder into your Pinecone index
(this needs to be done only once for indexing)
```bash
python load_pdfs.py
```

### 7. Run the FastAPI Server
uvicorn main:app --reload




