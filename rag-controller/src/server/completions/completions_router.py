from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from src.server.chat.chat_router import ChatBody, chat_completion
from src.server.utils.auth import authenticated

completions_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CompletionsBody(BaseModel):
    prompt: str
    system_prompt: str | None = None
    use_context: bool = False
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "How do you fry an egg?",
                    "system_prompt": "You are a rapper. Always answer with a rap.",
                    "stream": False,
                    "use_context": False,
                }
            ]
        }
    }


@completions_router.post(
    "/completions",
    response_model=None,
    summary="Completion",
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)
def prompt_completion(
    request: Request, body: CompletionsBody
) -> OpenAICompletion | StreamingResponse:
    """We recommend most users use our Chat completions API.

    Given a prompt, the model will return one predicted completion.

    Optionally include a `system_prompt` to influence the way the LLM answers.

    If `use_context` is set to `true`, the model will use context coming/

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):

    """
    messages = [OpenAIMessage(content=body.prompt, role="user")]
    # If system prompt is passed, create a fake message with the system prompt.
    if body.system_prompt:
        messages.insert(0, OpenAIMessage(content=body.system_prompt, role="system"))

    chat_body = ChatBody(
        messages=messages,
        use_context=body.use_context,
        stream=body.stream,
    )
    return chat_completion(request, chat_body)
