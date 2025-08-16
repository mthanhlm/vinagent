import pandas as pd
from datasets import load_dataset
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from aucodb.vectordb.factory import VectorDatabaseFactory
from aucodb.vectordb.processor import DocumentProcessor
from typing import Literal, Any, Dict, List
from vinagent.register import primary_function


class SearchLegalEngine:
    def __init__(
        self,
        top_k: int = 5,
        temp_data_path: Path = Path("data/test.parquet"),
        db_type: Literal[
            "chroma", "faiss", "milvus", "pgvector", "pinecone", "qdrant", "weaviate"
        ] = "milvus",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.top_k = top_k
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.temp_data_path = temp_data_path
        self.db_type = db_type
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.doc_processor = DocumentProcessor(splitter=self.text_splitter)

    def _download_legal_case(self):
        dataset = load_dataset(
            "joelniklaus/legal_case_document_summarization", split="test"
        )
        dataset.to_parquet(self.temp_data_path)

    def _create_legal_cases_data(self):
        self._download_legal_case()
        legal_case = pd.read_parquet(self.temp_data_path)

        self.docs = []
        for i, doc in legal_case.iterrows():
            doc = Document(
                page_content=doc["judgement"],
                metadata={
                    "judgement_case": i,
                    "dataset_name": doc["dataset_name"],
                    "summary": doc["summary"],
                },
            )
            self.docs.append(doc)
        return self.docs

    def _initialize_document_processor(self):
        self.vectordb_factory = VectorDatabaseFactory(
            db_type=self.db_type,
            embedding_model=self.embedding_model,
            doc_processor=self.doc_processor,
        )
        self._create_legal_cases_data()
        self.vectordb_factory.store_documents(self.docs)

    def exact_match_score(self, query, doc):
        # Convert strings to sets of words (case-insensitive, removing punctuation)
        query_words = set(query.lower().split())
        doc_words = set(doc.lower().split())

        # Calculate intersection of words
        common_words = query_words.intersection(doc_words)

        # Avoid division by zero
        if len(query_words) == 0 or len(doc_words) == 0:
            return 0.0

        # Calculate score: 0.5 * (|V_q ∩ V_d|/|V_q| + |V_q ∩ V_d|/|V_d|)
        score = 0.5 * (
            len(common_words) / len(query_words) + len(common_words) / len(doc_words)
        )

        return score

    def exact_match_search_query(self, query, docs):
        # Calculate scores for all documents
        scores = [
            (id_doc, self.exact_match_score(query, doc.page_content))
            for (id_doc, doc) in enumerate(docs)
        ]
        return scores

    def semantic_search_query(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        actual_top_k = top_k or self.top_k
        if self.vectordb_factory.vectordb.vector_store is None:
            raise ValueError("Vector store not initialized. Store documents first.")

        # Generate embedding for query
        query_vector = self.vectordb_factory.vectordb.embedding_model.embed_query(query)

        # Perform similarity search
        results = self.vectordb_factory.vectordb.client.search(
            collection_name=self.vectordb_factory.vectordb.collection_name,
            data=[query_vector],
            limit=actual_top_k,
            output_fields=["text"],
            search_params={
                "metric_type": self.vectordb_factory.vectordb.metric_type
            },  # Use consistent metric type
        )[0]
        returned_docs = [(doc.id, doc.distance) for doc in results]
        returned_docs = sorted(
            [doc for doc in returned_docs], key=lambda x: x[0], reverse=False
        )
        return returned_docs

    def query_fusion_score(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        w_semantic: float = 0.5,
    ):
        """Query a list of documents based on exact matching and semantic scores. Return a list of similar documents.
        Args:
            query (str): The query to search for.
            top_k (int): The number of documents to return. Defaults to self.top_k.
            threshold (float): The minimum fusion score to return. Defaults to None.
            w_semantic (float): The weight of the semantic score. Defaults to 0.5.
        Returns:
            list: A list of similar documents.
        """
        exact_match_scores = self.exact_match_search_query(query=query, docs=self.docs)
        semantic_scores = self.semantic_search_query(query=query, top_k=len(self.docs))
        scores = [
            (
                id_exac,
                {
                    "semantic_score": seman_score,
                    "exac_score": exac_score,
                    "fusion_score": (1 - w_semantic) * exac_score
                    + w_semantic * seman_score,
                },
            )
            for ((id_exac, exac_score), (id_seman, seman_score)) in list(
                zip(exact_match_scores, semantic_scores)
            )
        ]
        sorted_scores = sorted(
            scores, key=lambda x: x[1]["fusion_score"], reverse=True
        )[: min(top_k, len(self.docs))]
        sorted_docs = [(self.docs[i], score) for (i, score) in sorted_scores]
        if threshold:
            filter_docs = [
                doc for (doc, score) in sorted_docs if score["fusion_score"] > threshold
            ]
            return filter_docs
        else:
            return sorted_docs


@primary_function
def query_similar_legal_cases(
    query: str, n_legal_cases: int = 2, threshold: float = 0.6
):
    """Query the similar legal cases to the given query.
    Args:
        query (str): The query string.
        n_legal_cases (int): The number of legal cases
        threshold (float): The similarity threshold. Defaults to 0.6.

    Returns:
        The similar legal cases.
    """
    search_legal_engine = SearchLegalEngine(
        top_k=n_legal_cases,
        temp_data_path=Path("data/test.parquet"),
        db_type="milvus",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    search_legal_engine._create_legal_cases_data()
    search_legal_engine._initialize_document_processor()
    docs = search_legal_engine.query_fusion_score(
        query, top_k=n_legal_cases, threshold=threshold, w_semantic=0.7
    )
    return docs
