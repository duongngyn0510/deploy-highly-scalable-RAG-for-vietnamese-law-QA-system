import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.retrievers import BaseRetriever

from src.paths import local_data_path
from src.settings.settings import Settings


logger = logging.getLogger(__name__)


@singleton
class VectorStoreComponent:
    vector_store: VectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.vectorstore.database:
            case "postgres":
                try:
                    from llama_index.vector_stores.postgres import (  # type: ignore
                        PGVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras vector-stores-postgres`"
                    ) from e

                if settings.postgres is None:
                    raise ValueError(
                        "Postgres settings not found. Please provide settings."
                    )

                self.vector_store = typing.cast(
                    VectorStore,
                    PGVectorStore.from_params(
                        **settings.postgres.model_dump(exclude_none=True),
                        table_name="embeddings",
                        embed_dim=settings.embedding.embed_dim,
                    ),
                )

            case "chroma":
                try:
                    import chromadb  # type: ignore
                    from chromadb.config import (  # type: ignore
                        Settings as ChromaSettings,
                    )

                    from src.components.vector_store.batched_chroma import (
                        BatchedChromaVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "ChromaDB dependencies not found, install with `poetry install --extras vector-stores-chroma`"
                    ) from e

                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(
                    path=str((local_data_path / "chroma_db").absolute()),
                    settings=chroma_settings,
                )
                chroma_collection = chroma_client.get_or_create_collection(
                    "make_this_parameterizable_per_api_call"
                )  # TODO

                self.vector_store = typing.cast(
                    VectorStore,
                    BatchedChromaVectorStore(
                        chroma_client=chroma_client, chroma_collection=chroma_collection
                    ),
                )

            case "qdrant":
                try:
                    from llama_index.vector_stores.qdrant import (  # type: ignore
                        QdrantVectorStore,
                    )
                    from qdrant_client import QdrantClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
                    ) from e

                if settings.qdrant is None:
                    logger.info(
                        "Qdrant config not found. Using default settings."
                        "Trying to connect to Qdrant at localhost:6333."
                    )
                    client = QdrantClient()
                else:
                    client = QdrantClient(
                        **settings.qdrant.model_dump(exclude_none=True)
                    )
                self.vector_store = typing.cast(
                    VectorStore,
                    QdrantVectorStore(
                        client=client,
                        collection_name="make_this_parameterizable_per_api_call",
                    ),  # TODO
                )

            case "weaviate":
                try:
                    from src.components.vector_store.custom.weaviate.base import (  # type: ignore
                        CustomWeaviateVectorStore,
                    )
                    import weaviate  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Weaviate dependencies not found, install with `poetry install --extras vector-stores-weaviate` \
                            or implement it"
                    ) from e

                weaviate_settings = settings.weaviate

                client = weaviate.Client(
                    url=weaviate_settings.weaviate_endpoint,
                    auth_client_secret=weaviate.auth.AuthApiKey(
                        weaviate_settings.api_key
                    ),
                )

                self.vector_store = CustomWeaviateVectorStore(
                    weaviate_client=client, index_name=weaviate_settings.index_name
                )

            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Vectorstore database {settings.vectorstore.database} not supported"
                )

    def get_retriever(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:

        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

    def get_hybrid_retriever(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
    ) -> BaseRetriever:
        try:
            from src.components.vector_store.custom.retriever import HybridRetriever
            return HybridRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
            )
        except ImportError as e:
            raise ImportError(
                "Please implement HybridRetriever or import other custom Retriever"
            ) from e

    def close(self) -> None:
        if hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
