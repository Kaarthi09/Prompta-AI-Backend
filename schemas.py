from typing import List, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

# 1. Message Schema (Child)
class MessageRead(BaseModel):
    sender: str
    messageText: str
    createdAt: datetime
    messageIndex: int
    fileNames: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

# 2. Conversation Schema (Parent)
class ConversationRead(BaseModel):
    conversationId: Union[UUID, str]
    conversationName: str
    createdAt: datetime
    updatedAt: datetime
    isArchived: int
    isPinned: int
    messages: List[MessageRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

class RenameRequest(BaseModel):
    newTitle: str

class ChatResponse(BaseModel):
    conversationId: Union[UUID, str]
    messageIndex: int
    message: str