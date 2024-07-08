from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from src.server.retriever.retriever_service import RetrieverService
from src.server.utils.auth import authenticated

retriever_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])

class ChatBody(BaseModel):
    text: str

@retriever_router.post(
    "/retrieve",
)
def retrieve(request: Request, body: ChatBody) -> int:

    service = request.state.injector.get(RetrieverService)
    len_nodes = service.retrieve(body.text)
    return len_nodes
