import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from src.components.llm.custom.vllm.utils import get_response, post_http_request


class VllmServer(LLM):
    model: Optional[str] = Field(description="The HuggingFace Model to use.")

    temperature: float = Field(description="The temperature to use for sampling.")

    n: int = Field(
        default=1,
        description="Number of output sequences to return for the given prompt.",
    )

    best_of: Optional[int] = Field(
        default=None,
        description="Number of output sequences that are generated from the prompt.",
    )

    presence_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on whether they appear in the generated text so far.",
    )

    frequency_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on their frequency in the generated text so far.",
    )

    repetition_penalty: float = Field(
        default=0.0,
    )

    top_p: float = Field(
        default=1.0,
        description="Float that controls the cumulative probability of the top tokens to consider.",
    )

    top_k: int = Field(
        default=-1,
        description="Integer that controls the number of top tokens to consider.",
    )

    min_p: int = Field(default=0)

    use_beam_search: bool = Field(
        default=False, description="Whether to use beam search instead of sampling."
    )

    stop: Optional[List[str]] = Field(
        default=None,
        description="List of strings that stop the generation when they are generated.",
    )

    length_penalty: float = Field(
        default=1.0,
        description="List of strings that stop the generation when they are generated.",
    )

    early_stopping: bool = Field(
        default=False,
        description="List of strings that stop the generation when they are generated.",
    )

    ignore_eos: bool = Field(
        default=False,
        description="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    )

    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )

    logprobs: Optional[int] = Field(
        default=None,
        description="Number of log probabilities to return per output token.",
    )

    prompt_logprobs: Optional[int] = Field(
        default=None,
    )

    skip_special_tokens: bool = Field(
        default=None,
    )

    spaces_between_special_tokens: bool = Field(
        default=None,
    )

    api_url: str = Field(description="The api url for vllm server")

    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_url: str,
        model: str = "Viet-Mistral/Vistral-7B-Chat",
        temperature: float = 1.0,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: int = 0,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 512,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._client = None
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            api_url=api_url,
            model=model,
            temperature=temperature,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            use_beam_search=use_beam_search,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            stop=stop,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VllmServer"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:

        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "n": self.n,
            "best_of": self.best_of,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "use_beam_search": self.use_beam_search,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "max_tokens": self.max_tokens,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
        }
        return {**base_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def __del__(self) -> None:
        ...

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt

        response = post_http_request(self.api_url, sampling_params, stream=False)
        output = get_response(response)
        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=True)

        def gen() -> CompletionResponseGen:
            # response_str = ""
            prev_prefix_len = len(prompt)
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    increasing_concat = data["choices"][0]["text"]
                    pref = prev_prefix_len
                    prev_prefix_len = len(increasing_concat)
                    yield CompletionResponse(
                        text=increasing_concat, delta=increasing_concat[pref:]
                    )

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        return self.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # build sampling parameters
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt

        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, **kwargs):
                yield message

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        return self.chat(messages, **kwargs)
