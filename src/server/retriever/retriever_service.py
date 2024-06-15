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
        if self.settings.rag.custom_retriever.enabled:
            if self.settings.rag.custom_retriever.version == "v1":
                vector_index_retriever = self.vector_store_component.get_hybrid_retriever_v1(
                    index=self.index,
                    similarity_top_k=self.settings.rag.similarity_top_k,
                    retrieval_top_k=self.settings.hybrid_retriever_v1.retrieval_top_k,
                    dense_threshold=self.settings.hybrid_retriever_v1.dense_threshold,
                    bm25_threshold=self.settings.hybrid_retriever_v1.bm25_threshold,
                )
            elif self.settings.rag.custom_retriever.version == "v2":
                vector_index_retriever = self.vector_store_component.get_hybrid_retriever_v2(
                    index=self.index,
                    similarity_top_k=self.settings.rag.similarity_top_k,
                    retrieval_top_k=self.settings.hybrid_retriever_v2.retrieval_top_k,
                    dense_threshold=self.settings.hybrid_retriever_v2.dense_threshold,
                    bm25_threshold=self.settings.hybrid_retriever_v2.bm25_threshold,
                    alpha=self.settings.hybrid_retriever_v2.alpha,
                )
            else:
                raise ValueError(
                    f"Hybrid retriever {self.settings.rag.custom_retriever.version} not implement"
                )

        else:
            vector_index_retriever = self.vector_store_component.get_retriever(
                index=self.index,
                similarity_top_k=self.settings.rag.similarity_top_k,
            )

        nodes = vector_index_retriever.retrieve(text)
        if self.settings.translation.enabled:
            from src.server.chat.custom_chat_engine import (
                CustomChatEngineWithTranslation,
            )
            from src.components.translation.translation_component import (
                TranslationComponent,
            )

            # context_str = "\n\n".join(
            #     [
            #         self._translation(
            #             n.node.get_content(metadata_mode=MetadataMode.LLM).strip(),
            #             forward=True
            #         )
            #         for n in nodes
            #     ]
            # )
        return len(nodes)

        #     node_postprocessors.append(rerank_postprocessor)

    def retrieve(self, text: str) -> int:
        len_nodes = self._retrieve(text)
        return len_nodes
