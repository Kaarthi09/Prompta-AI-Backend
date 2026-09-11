from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
import uuid

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class ConversationModel(BaseModel):
    """
    Pydantic Model for Conversations collection.
    Enforces data integrity, defaults, and schema structure for MongoDB documents.
    """
    conversationId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversationName: str = Field(..., max_length=200)
    isArchived: int = Field(default=0, ge=0, le=1)
    isPinned: int = Field(default=0, ge=0, le=1)
    createdAt: datetime = Field(default_factory=utcnow)
    updatedAt: datetime = Field(default_factory=utcnow)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    def to_mongo(self) -> Dict[str, Any]:
        """Convert model to dictionary suitable for PyMongo insertion."""
        return self.model_dump()

    @classmethod
    def from_mongo(cls, doc: Dict[str, Any]) -> "ConversationModel":
        """Construct model instance from MongoDB document."""
        if not doc:
            return None
        doc_copy = dict(doc)
        doc_copy.pop("_id", None)
        # Ensure UUID fields are strings
        if isinstance(doc_copy.get("conversationId"), uuid.UUID):
            doc_copy["conversationId"] = str(doc_copy["conversationId"])
        return cls(**doc_copy)


class MessageModel(BaseModel):
    """
    Pydantic Model for Messages collection.
    Enforces data integrity for individual user and bot chat messages.
    """
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversationId: str
    sender: str = Field(..., max_length=50)
    messageText: str
    messageIndex: int
    createdAt: datetime = Field(default_factory=utcnow)
    fileNames: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    def to_mongo(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo(cls, doc: Dict[str, Any]) -> "MessageModel":
        if not doc:
            return None
        doc_copy = dict(doc)
        doc_copy.pop("_id", None)
        if isinstance(doc_copy.get("messageId"), uuid.UUID):
            doc_copy["messageId"] = str(doc_copy["messageId"])
        if isinstance(doc_copy.get("conversationId"), uuid.UUID):
            doc_copy["conversationId"] = str(doc_copy["conversationId"])
        return cls(**doc_copy)


class FileChunkModel(BaseModel):
    """
    Pydantic Model for FileChunks collection.
    Enforces data integrity for ingested document chunks.
    """
    chunkId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fileName: str
    chunkIndex: int
    content: str
    conversationId: Optional[str] = None
    messageIndex: Optional[int] = None
    createdAt: datetime = Field(default_factory=utcnow)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    def to_mongo(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo(cls, doc: Dict[str, Any]) -> "FileChunkModel":
        if not doc:
            return None
        doc_copy = dict(doc)
        doc_copy.pop("_id", None)
        if isinstance(doc_copy.get("conversationId"), uuid.UUID):
            doc_copy["conversationId"] = str(doc_copy["conversationId"])
        return cls(**doc_copy)