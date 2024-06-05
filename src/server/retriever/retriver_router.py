from fastapi import APIRouter, Depends, Request
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
    to_openai_response,
    to_openai_sse_stream,
)
from src.server.retriever.retriever_service import RetrieverService
from src.server.utils.auth import authenticated

retriever_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChatBody(BaseModel):
    text: str
    # messages: list[OpenAIMessage]
    # use_context: bool = False
    # stream: bool = False

    # model_config = {
    #     "json_schema_extra": {
    #         "examples": [
    #             {
    #                 "messages": [
    #                     {
    #                         "role": "system",
    #                         "content": "You are a rapper. Always answer with a rap.",
    #                     },
    #                     {
    #                         "role": "user",
    #                         "content": "How do you fry an egg?",
    #                     },
    #                 ],
    #                 "stream": False,
    #                 "use_context": True,
    #             }
    #         ]
    #     }
    # }


@retriever_router.post(
    "/retrieve",
)
def retrieve(request: Request, body: ChatBody) -> int:

    service = request.state.injector.get(RetrieverService)
    len_nodes = service.retrieve(body.text)
    return len_nodes
