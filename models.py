from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None 

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]
