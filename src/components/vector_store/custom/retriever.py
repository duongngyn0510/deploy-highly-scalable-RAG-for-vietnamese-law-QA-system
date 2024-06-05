from typing import List, Dict

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle


class HybridRetrieverV1(BaseRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int,
        retrieval_top_k: int,
        dense_threshold: float = 0.35,
        bm25_threshold: float = 8.0,
    ):
        self.similarity_top_k = similarity_top_k
        self.retrieval_top_k = retrieval_top_k
        self.dense_threshold = dense_threshold
        self.bm25_threshold = bm25_threshold

        self.dense_retrieval = index.as_retriever(
            similarity_top_k=self.similarity_top_k, vector_store_query_mode="default"
        )

        self.bm25_retrieval = index.as_retriever(
            similarity_top_k=self.similarity_top_k, vector_store_query_mode="sparse"
        )
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, **kwargs):
        """
        Query should be segmented and refined
        """
        import time

        x = time.time()
        vector_nodes: List[NodeWithScore] = self.dense_retrieval.retrieve(
            query_bundle, **kwargs
        )
        print("vector_nodes time", time.time() - x)
        x = time.time()
        bm25_nodes: List[NodeWithScore] = self.bm25_retrieval.retrieve(
            query_bundle, **kwargs
        )
        print("bm25_nodes time", time.time() - x)
        all_nodes: Dict[str, NodeWithScore] = {}

        for vector_node in vector_nodes:
            vector_node.metadata["dense_score"] = vector_node.score
            vector_node.metadata["bm25_score"] = 0.0
            all_nodes[vector_node.node_id] = vector_node

        for bm25_node in bm25_nodes:
            bm25_node_id = bm25_node.node_id
            if bm25_node_id in all_nodes:
                vector_node = all_nodes[bm25_node_id]
                bm25_node.metadata["dense_score"] = vector_node.metadata["dense_score"]
            else:
                bm25_node.metadata["dense_score"] = 0.0

            bm25_node.metadata["bm25_score"] = bm25_node.score
            all_nodes[bm25_node.node_id] = bm25_node
        print("len(all_nodes)", len(all_nodes))

        return [
            node
            for node in all_nodes.values()
            if node.metadata["bm25_score"] > self.bm25_threshold
            and node.metadata["dense_score"] > self.dense_threshold
        ][: self.retrieval_top_k]


class HybridRetrieverV2(BaseRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int,
        retrieval_top_k: int,
        dense_threshold: float = 0.15,
        bm25_threshold: float = 6.0,
        alpha: float = 0.8,
    ):
        self.similarity_top_k = similarity_top_k
        self.retrieval_top_k = retrieval_top_k
        self.dense_threshold = dense_threshold
        self.bm25_threshold = bm25_threshold
        self.alpha = alpha
        self.dense_retrieval = index.as_retriever(
            similarity_top_k=self.similarity_top_k, vector_store_query_mode="default"
        )

        self.bm25_retrieval = index.as_retriever(
            similarity_top_k=self.similarity_top_k, vector_store_query_mode="sparse"
        )
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, **kwargs):
        """
        Query should be segmented and refined
        """
        import time

        x = time.time()
        vector_nodes: List[NodeWithScore] = self.dense_retrieval.retrieve(
            query_bundle, **kwargs
        )
        print("vector_nodes time", time.time() - x)
        x = time.time()
        bm25_nodes: List[NodeWithScore] = self.bm25_retrieval.retrieve(
            query_bundle, **kwargs
        )
        print("bm25_nodes time", time.time() - x)
        all_nodes: Dict[str, NodeWithScore] = {}

        for vector_node in vector_nodes:
            vector_node.metadata["dense_score"] = vector_node.score
            vector_node.metadata["bm25_score"] = 0.0
            all_nodes[vector_node.node_id] = vector_node

        for bm25_node in bm25_nodes:
            bm25_node_id = bm25_node.node_id
            if bm25_node_id in all_nodes:
                vector_node = all_nodes[bm25_node_id]
                bm25_node.metadata["dense_score"] = vector_node.metadata["dense_score"]
            else:
                bm25_node.metadata["dense_score"] = 0.0
            bm25_node.metadata["bm25_score"] = bm25_node.score
            all_nodes[bm25_node.node_id] = bm25_node

        max_dense_score = -9999
        min_dense_score = 9999
        max_bm25_score = -9999
        min_bm25_score = 9999

        for node in all_nodes.values():
            max_dense_score = max(max_dense_score, node.metadata["dense_score"])
            min_dense_score = min(min_dense_score, node.metadata["dense_score"])
            max_bm25_score = max(max_bm25_score, node.metadata["bm25_score"])
            min_bm25_score = min(min_bm25_score, node.metadata["bm25_score"])

        for node in all_nodes.values():
            bm25_score = (node.metadata["bm25_score"] - min_bm25_score) / max_bm25_score
            dense_score = (
                node.metadata["dense_score"] - min_dense_score
            ) / max_dense_score
            node.score = self.alpha * dense_score + (1 - self.alpha) * bm25_score

        all_nodes = sorted(all_nodes.values(), key=lambda x: -x.score)
        x = [
            node
            for node in all_nodes
            if node.metadata["bm25_score"] > self.bm25_threshold
            and node.metadata["dense_score"] > self.dense_threshold
        ][: self.retrieval_top_k]
        print("len(x)", len(x))
        return [
            node
            for node in all_nodes
            if node.metadata["bm25_score"] > self.bm25_threshold
            and node.metadata["dense_score"] > self.dense_threshold
        ][: self.retrieval_top_k]
