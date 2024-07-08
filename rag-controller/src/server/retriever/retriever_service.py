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
from src.components.rerank.rerank_component import RerankComponent
from src.settings.settings import Settings


@singleton
class RetrieverService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        rerank_component: RerankComponent, 
    ) -> None:
        self.settings = settings
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.rerank_component = rerank_component
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store_component.vector_store,
            embed_model=embedding_component.embedding_model,
        )

    def _retrieve(self, text: str) -> int:
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

        nodes = vector_index_retriever.retrieve(text)
        return len(nodes)

    def retrieve(self, text: str) -> int:
        len_nodes = self._retrieve(text)
        return len_nodes
