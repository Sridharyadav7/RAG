from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
app = FastAPI(title="RAG Medical API")

# Global variables to store components
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain

    try:
        print("Initializing RAG components...")
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=1024,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
        
        vectorstore = PineconeVectorStore(
            embedding=embedding_model, 
            index_name=PINECONE_INDEX_NAME
        )
        
        retriever = vectorstore.as_retriever(search_type="similarity", k=1)
        
        prompt_template = """
        You are a helpful and professional medical assistant speaking to a patient on a live call.
        Use the retrieved context to provide a short, accurate, and human-like answer to the patient’s medical question.
        Keep the response clear and easy to understand — avoid complex language or unnecessary details.
        Keep it concise — aim for 1–2 short paragraphs suitable for a live conversation.
        If the information is not available in the context or you're unsure, clearly say: “I'm not sure about that.”

        Context: {context}

        Question: {question}

        Helpful Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=retriever, 
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RAG components initialized successfully")
        
    except Exception as e:
        print(f"Error initializing RAG components: {str(e)}")
        import traceback
        traceback.print_exc()

# ----------- VAPI Tool-Call-Compatible /ask Route -----------

class FunctionDetails(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    id: str
    function: FunctionDetails

class VapiMessage(BaseModel):
    toolCallList: List[ToolCall]

class VapiRequest(BaseModel):
    message: VapiMessage

from fastapi import Request  # Add this import

@app.post("/ask")
async def ask_question(request: Request):
    global rag_chain

    # Step 1: Print raw incoming body
    raw_body = await request.body()
    print("🔍 Raw Request Body from Vapi:\n", raw_body.decode())

    # Step 2: Try to parse JSON manually for debugging
    try:
        json_data = await request.json()
        print("📦 Parsed JSON:\n", json_data)
    except Exception as e:
        print("❌ Failed to parse JSON body:", str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Step 3: Validate with your original Pydantic model
    try:
        parsed_request = VapiRequest(**json_data)
    except Exception as e:
        print("⚠️ Pydantic Validation Error:", str(e))
        raise HTTPException(status_code=422, detail="Invalid tool-call format")

    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please check server logs.")

    try:
        tool_call = parsed_request.message.toolCallList[0]
        query_text = tool_call.function.arguments["query"]


        result = await asyncio.to_thread(rag_chain.invoke, {"query": query_text})
        
        raw_answer = result.get("result", "No answer generated")
        cleaned_answer = raw_answer.replace("\n", " ").replace("*", "").replace("•", "").strip()

        return {
            "results": [
                {
                    "toolCallId": tool_call.id,
                    "result": cleaned_answer
                }
            ]
        }

    except Exception as e:
        import traceback
        print("🔥 Internal Server Error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


    except Exception as e:
        import traceback
        err_details = traceback.format_exc()
        print(f"Error in /ask endpoint: {str(e)}\n{err_details}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# ------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_initialized": rag_chain is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
