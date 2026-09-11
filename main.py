from datetime import datetime, timezone
from contextlib import asynccontextmanager
import os
import shutil
import uuid
import warnings
import logging
from typing import List, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from pydantic import BaseModel
import requests
import textwrap

# MongoDB & Database imports
from fastapi.middleware.cors import CORSMiddleware
from database import (
    get_db, 
    conversations_collection, 
    messages_collection, 
    file_chunks_collection,
    init_db_indexes
)
from models import ConversationModel, MessageModel, FileChunkModel, utcnow
from schemas import ConversationRead, MessageRead, RenameRequest, ChatResponse

# --- LangChain & Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, 
    PDFPlumberLoader
)
from langchain_core.documents import Document

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploaded_docs"
VECTOR_DB_PATH = "vector_store/faiss_index"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# --- Global State ---
print("--- STARTUP: Loading Embedding Model... ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = None

def load_vector_store():
    """Attempts to load the existing FAISS index from disk."""
    global vector_store
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("--- STARTUP: Loaded existing FAISS index. ---")
        except Exception as e:
            print(f"--- STARTUP: Could not load existing index: {e} ---")
            vector_store = None
    else:
        print("--- STARTUP: No existing index found. Waiting for uploads. ---")
        vector_store = None

# Load vector store on startup
load_vector_store()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MongoDB collections and indexes on server startup
    init_db_indexes()
    try:
        yield
    finally:
        # Shutdown cleanup: remove uploaded files and persisted vector DB
        global vector_store
        try:
            if os.path.exists(UPLOAD_DIR):
                for name in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove upload path {path}: {e}")

            if os.path.exists(VECTOR_DB_PATH):
                for name in os.listdir(VECTOR_DB_PATH):
                    path = os.path.join(VECTOR_DB_PATH, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove vector store path {path}: {e}")
            vector_store = None
            logger.info("Lifespan cleanup complete.")
        except Exception as e:
            logger.error(f"Error during lifespan cleanup: {e}")

app = FastAPI(title="Prompta AI Backend", lifespan=lifespan)

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_pdf(file_path: str) -> List[Document]:
    """Helper to parse PDF files into LangChain Documents."""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.warning(f"PDFPlumber failed for {file_path}: {e}. Falling back to Unstructured...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()
        except Exception as ex:
            logger.error(f"Failed to parse PDF {file_path}: {ex}")
            return []

def update_vector_db(new_docs: List[Document]):
    """Chunks documents and updates the global FAISS index."""
    global vector_store
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(new_docs)
    
    if not chunks:
        return

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)
    
    vector_store.save_local(VECTOR_DB_PATH)
    logger.info(f"Vector store updated with {len(chunks)} new chunks.")

# --- API Input Models ---
class QueryRequest(BaseModel):
    question: str
    model_name: str = "gemma3:1b"
    conversation_id: Optional[Union[uuid.UUID, str]] = None

# --- Management Endpoints (MongoDB + Pydantic) ---

@app.get("/chat/getAllConversations", response_model=List[ConversationRead])
def get_all_conversations():
    """
    Fetches all conversations with their messages, sorted by updatedAt Descending.
    Uses Pydantic models for MongoDB document deserialization and validation.
    """
    # 1. Fetch conversations sorted by updatedAt descending
    convo_docs = list(conversations_collection.find().sort("updatedAt", -1))
    results = []

    for c_doc in convo_docs:
        convo_model = ConversationModel.from_mongo(c_doc)
        if not convo_model:
            continue

        # 2. Fetch associated messages sorted by messageIndex
        msg_docs = list(messages_collection.find({"conversationId": convo_model.conversationId}).sort("messageIndex", 1))
        message_reads = []

        for m_doc in msg_docs:
            msg_model = MessageModel.from_mongo(m_doc)
            if not msg_model:
                continue

            # Fetch file names linked to this message chunk
            chunk_docs = list(file_chunks_collection.find({
                "conversationId": convo_model.conversationId,
                "messageIndex": msg_model.messageIndex
            }))
            filenames = [ch["fileName"] for ch in chunk_docs if "fileName" in ch]
            dedup_filenames = list(dict.fromkeys(filenames)) if filenames else []

            msg_read = MessageRead(
                sender=msg_model.sender,
                messageText=msg_model.messageText,
                createdAt=msg_model.createdAt,
                messageIndex=msg_model.messageIndex,
                fileNames=dedup_filenames
            )
            message_reads.append(msg_read)

        convo_read = ConversationRead(
            conversationId=convo_model.conversationId,
            conversationName=convo_model.conversationName,
            createdAt=convo_model.createdAt,
            updatedAt=convo_model.updatedAt,
            isArchived=convo_model.isArchived,
            isPinned=convo_model.isPinned,
            messages=message_reads
        )
        results.append(convo_read)

    return results

@app.get("/chat/getConversation/{conversation_id}", response_model=ConversationRead)
def get_conversation(conversation_id: str):
    cid_str = str(conversation_id)
    c_doc = conversations_collection.find_one({"conversationId": cid_str})
    if not c_doc:
        raise HTTPException(status_code=404, detail="Conversation not found")

    convo_model = ConversationModel.from_mongo(c_doc)
    msg_docs = list(messages_collection.find({"conversationId": cid_str}).sort("messageIndex", 1))
    message_reads = []

    for m_doc in msg_docs:
        msg_model = MessageModel.from_mongo(m_doc)
        if not msg_model:
            continue

        chunk_docs = list(file_chunks_collection.find({
            "conversationId": cid_str,
            "messageIndex": msg_model.messageIndex
        }))
        filenames = [ch["fileName"] for ch in chunk_docs if "fileName" in ch]
        dedup_filenames = list(dict.fromkeys(filenames)) if filenames else []

        msg_read = MessageRead(
            sender=msg_model.sender,
            messageText=msg_model.messageText,
            createdAt=msg_model.createdAt,
            messageIndex=msg_model.messageIndex,
            fileNames=dedup_filenames
        )
        message_reads.append(msg_read)

    return ConversationRead(
        conversationId=convo_model.conversationId,
        conversationName=convo_model.conversationName,
        createdAt=convo_model.createdAt,
        updatedAt=convo_model.updatedAt,
        isArchived=convo_model.isArchived,
        isPinned=convo_model.isPinned,
        messages=message_reads
    )

@app.delete("/chat/deleteById/{conversation_id}")
def delete_conversation(conversation_id: str):
    cid_str = str(conversation_id)
    res = conversations_collection.delete_one({"conversationId": cid_str})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages_collection.delete_many({"conversationId": cid_str})
    file_chunks_collection.delete_many({"conversationId": cid_str})
    return {"message": "Conversation deleted."}

@app.delete("/chat/deleteAll")
def delete_all_conversations():
    conversations_collection.delete_many({})
    messages_collection.delete_many({})
    file_chunks_collection.delete_many({})
    return {"message": "All conversations deleted."}

@app.put("/chat/renameConversation/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest):
    cid_str = str(conversation_id)
    now = utcnow()
    res = conversations_collection.update_one(
        {"conversationId": cid_str},
        {"$set": {"conversationName": req.newTitle, "updatedAt": now}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation renamed."}

@app.put("/chat/Archive/{conversation_id}")
def archive_conversation(conversation_id: str):
    cid_str = str(conversation_id)
    convo = conversations_collection.find_one({"conversationId": cid_str})
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_val = 1 if convo.get("isArchived", 0) == 0 else 0
    conversations_collection.update_one(
        {"conversationId": cid_str},
        {"$set": {"isArchived": new_val, "updatedAt": utcnow()}}
    )
    return {"conversationId": cid_str, "isArchived": new_val}

@app.put("/chat/pin/{conversation_id}")
def pin_conversation(conversation_id: str):
    cid_str = str(conversation_id)
    convo = conversations_collection.find_one({"conversationId": cid_str})
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_val = 1 if convo.get("isPinned", 0) == 0 else 0
    conversations_collection.update_one(
        {"conversationId": cid_str},
        {"$set": {"isPinned": new_val, "updatedAt": utcnow()}}
    )
    return {"conversationId": cid_str, "isPinned": new_val}

# --- AI & Ingestion Endpoints ---

def process_and_chunk_pdf(file_path: str) -> List[Document]:
    """Loads PDF and returns chunks."""
    raw_docs = process_pdf(file_path)
    if not raw_docs:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(raw_docs)

@app.post("/ingest", summary="Upload PDF files to Knowledge Base")
async def ingest_files(
    files: List[UploadFile] = File(...), 
    conversation_id: Optional[str] = Form(None), 
    message_index: int = Form(0)
):
    global vector_store
    processed_count = 0
    total_new_chunks = 0
    
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        chunks = process_and_chunk_pdf(file_location)
        
        if chunks:
            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                vector_store.add_documents(chunks)
            
            for i, chunk in enumerate(chunks):
                # Pydantic model enforces chunk data structure & type safety
                db_chunk = FileChunkModel(
                    fileName=file.filename,
                    chunkIndex=i,
                    content=chunk.page_content,
                    conversationId=str(conversation_id) if conversation_id else None,
                    messageIndex=message_index 
                )
                file_chunks_collection.insert_one(db_chunk.to_mongo())
            
            processed_count += 1
            total_new_chunks += len(chunks)
            
    if processed_count == 0:
        return {"message": "No valid text extracted."}
        
    return {"message": f"Ingested {processed_count} files.", "chunks_added": total_new_chunks}

@app.post("/ask", response_model=ChatResponse)
def ask_question(req: QueryRequest):
    """
    1. Retrieval (RAG)
    2. Generation (Ollama)
    3. Persistence (Save to MongoDB with Pydantic model validation)
    """
    global vector_store
    context_text = ""
    
    if vector_store:
        try:
            docs = vector_store.similarity_search(req.question, k=2)
            if docs:
                joined_docs = "\n\n".join([d.page_content for d in docs])
                context_text = textwrap.shorten(joined_docs, width=2000, placeholder=" ...")
        except Exception as e:
            logger.error(f"RAG Retrieval failed: {e}")

    print(f"--- RAG Context Retrieved: {context_text}")
    
    if context_text:
        prompt = (
            f"CONTEXT:\n{context_text}\n\n"
            f"Question: {req.question}\n"
            "INSTRUCTION: You are a helpful assistant."
            "Use the provided context to answer the question only if the Context is relevant to the Question."
            "If not relevant then ignore the context and answer based on your own knowledge."
        )
    else:
        prompt = (
            f"Question: {req.question}\n"
            "INSTRUCTION: Answer using your general knowledge."
        )

    payload = {
        "model": req.model_name,
        "prompt": prompt,
        "temperature": 0.3,
        "stream": False,
        "keep_alive": -1
    }
    
    bot_answer = ""
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        bot_answer = response.json().get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # --- DATABASE PERSISTENCE WITH PYDANTIC ---
    cid_str = str(req.conversation_id) if req.conversation_id else str(uuid.uuid4())
    c_doc = conversations_collection.find_one({"conversationId": cid_str})

    if not c_doc:
        convo_name = req.question[:50] + "..." if len(req.question) > 50 else req.question
        convo_model = ConversationModel(
            conversationId=cid_str,
            conversationName=convo_name,
            createdAt=utcnow(),
            updatedAt=utcnow()
        )
        conversations_collection.insert_one(convo_model.to_mongo())
    else:
        conversations_collection.update_one(
            {"conversationId": cid_str},
            {"$set": {"updatedAt": utcnow()}}
        )

    # Compute max messageIndex
    latest_msg = list(messages_collection.find({"conversationId": cid_str}).sort("messageIndex", -1).limit(1))
    current_max_index = latest_msg[0]["messageIndex"] if latest_msg else -1
    next_index = current_max_index + 1

    # Insert User message using Pydantic model validation
    user_msg = MessageModel(
        conversationId=cid_str,
        sender="User",
        messageText=req.question,
        messageIndex=next_index
    )
    messages_collection.insert_one(user_msg.to_mongo())

    # Insert Bot message using Pydantic model validation
    bot_msg = MessageModel(
        conversationId=cid_str,
        sender="Bot",
        messageText=bot_answer,
        messageIndex=next_index + 1
    )
    messages_collection.insert_one(bot_msg.to_mongo())

    return {
        "conversationId": cid_str,
        "messageIndex": next_index + 1,
        "message": bot_answer
    }

@app.post("/chat/loadContext/{conversation_id}")
def load_context_from_db(conversation_id: str):
    """
    Hydrates the Vector DB with chunks stored in MongoDB for a specific conversation.
    """
    global vector_store
    cid_str = str(conversation_id)

    chunk_docs = list(file_chunks_collection.find({"conversationId": cid_str}))

    if not chunk_docs:
        return {"message": "No documents found for this conversation.", "count": 0}

    documents = []
    for c_doc in chunk_docs:
        chunk_model = FileChunkModel.from_mongo(c_doc)
        if chunk_model:
            doc = Document(
                page_content=chunk_model.content,
                metadata={
                    "source": chunk_model.fileName,
                    "chunk_index": chunk_model.chunkIndex
                }
            )
            documents.append(doc)

    logger.info(f"Hydrating vector DB with {len(documents)} chunks from MongoDB...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return {"message": "Context loaded successfully", "count": len(documents)}

@app.post("/chat/clearContext")
def clear_context():
    """
    Clears the in-memory Vector DB.
    """
    global vector_store
    vector_store = None
    logger.info("Vector DB context cleared.")
    return {"message": "Context cleared"}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)