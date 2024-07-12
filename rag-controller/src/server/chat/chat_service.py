from dataclasses import dataclass

from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor


from llama_index.core.types import TokenGen
from pydantic import BaseModel

from src.components.embedding.embedding_component import EmbeddingComponent
from src.components.llm.llm_component import LLMComponent
from src.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from src.settings.settings import Settings
from src.server.chat.custom_chat_engine import CustomChatEngineWithTranslation
from src.components.translation.translation_component import TranslationComponent


class Completion(BaseModel):
    response: str


class CompletionGen(BaseModel):
    response: TokenGen


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store_component.vector_store,
            embed_model=embedding_component.embedding_model,
        )

    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
    ) -> BaseChatEngine:
        if use_context:
            if self.settings.rag.hybrid_retriever.enabled:
                vector_index_retriever = self.vector_store_component.get_hybrid_retriever(
                    index=self.index,
                    similarity_top_k=self.settings.rag.similarity_top_k,
                )

            else:
                vector_index_retriever = self.vector_store_component.get_retriever(
                    index=self.index,
                    similarity_top_k=self.settings.rag.similarity_top_k,
                )

            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                SimilarityPostprocessor(
                    similarity_cutoff=self.settings.rag.similarity_value
                ),
            ]

            # Is use rerank ?
            if self.settings.rag.rerank.enabled:
                from src.components.rerank.rerank_component import RerankComponent

                rerank_postprocessor = RerankComponent.rerank
                node_postprocessors.append(rerank_postprocessor)

            # Is use automatically redirect requests ?
            if self.settings.auto_redirect.enabled:
                try:
                    from src.components.llm.custom.nvidia_nim.base import NvidiaNim
                    from prometheus_api_client import PrometheusConnect 
                except ImportError as e:
                    raise e
                
                prometheus_url = self.settings.auto_redirect.prometheus_url
                prometheus_query = self.settings.auto_redirect.prometheus_query
                threshold = self.settings.auto_redirect.threshold

                # Setup prometheus
                prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
                result = prom.custom_query(query=prometheus_query)
                print("Connect prom")
                total_requests = float(result[0]['value'][1])
                
                if total_requests > threshold:
                    nvidia_nim_settings = self.settings.nvidia_nim
                    self.llm = NvidiaNim(
                        model=nvidia_nim_settings.model,
                        temperature=nvidia_nim_settings.temperature,
                        top_p=nvidia_nim_settings.top_p,
                        max_tokens=nvidia_nim_settings.max_tokens,
                        api_key=nvidia_nim_settings.api_key,
                        api_base=nvidia_nim_settings.api_base,
                    )
                else:
                    self.llm = self.llm_component.llm
            else:
                self.llm = self.llm_component.llm

            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=vector_index_retriever,
                llm=self.llm,
                node_postprocessors=node_postprocessors,
                verbose=True,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                llm=self.llm_component.llm,
            )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )
        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
        )
        response = chat_engine.stream_chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        completion_gen = CompletionGen(
            response=response.response_gen, 
        )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        completion = Completion(response=wrapped_response.response)
        return completion