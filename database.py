import os
from pymongo import MongoClient, ASCENDING
from pymongo.database import Database

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "PromptaAIDB")

# Create PyMongo Client
client = MongoClient(MONGODB_URI)
db: Database = client[DB_NAME]

# Collections
conversations_collection = db["conversations"]
messages_collection = db["messages"]
file_chunks_collection = db["file_chunks"]

def init_db_indexes():
    """Ensure database indexes exist for performance and unique constraint integrity."""
    try:
        conversations_collection.create_index([("conversationId", ASCENDING)], unique=True)
        messages_collection.create_index([("conversationId", ASCENDING), ("messageIndex", ASCENDING)])
        file_chunks_collection.create_index([("conversationId", ASCENDING), ("messageIndex", ASCENDING)])
        print(f"--- MongoDB initialized successfully. Connected to DB '{DB_NAME}' ---")
    except Exception as e:
        print(f"--- Error initializing MongoDB indexes: {e} ---")

def get_db() -> Database:
    """FastAPI Dependency for obtaining the MongoDB database instance."""
    return db