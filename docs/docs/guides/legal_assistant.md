# Legal Assistant with vinagent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/tutorials/guides/4.Legal_Assistant.ipynb)

This tutorial demonstrates how to build a comprehensive legal assistant using vinagent that can help attorneys and legal professionals with various tasks including:

- Finding similar legal cases using semantic and exact-match search
- Summarizing legal cases for quick review
- Creating timelines of legal events
- Analyzing arguments and their strengths/weaknesses
- Conducting jurisdictional analysis
- Evaluating ethical considerations in court rulings

## Prerequisite Installation

Install the necessary dependencies:


```python
%pip install vinagent==0.0.4.post7 datasets==4.0.0
```

Next, setup environment variables by creating a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

You can access [OpenAI](https://platform.openai.com/api-keys) site to create a free OpenAI key, which allows to use `GPT-4o-mini`.

## Prepare Data and Tool

Let's download a legal case example dataset from huggingface. This dataset comprises 200 legal cases for `test` and 7777 legal cases for `train` datasets. To simplify the demo on local, we only use test dataset.


```python
from datasets import load_dataset

dataset = load_dataset("joelniklaus/legal_case_document_summarization", split="test")
dataset.to_parquet("data/test.parquet")
```

    Creating parquet from Arrow format: 100%|██████████| 1/1 [00:00<00:00,  8.60ba/s]


```python
import pandas as pd
legal_case = pd.read_parquet("data/test.parquet")
legal_case.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>judgement</th>
      <th>dataset_name</th>
      <th>summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Appeal No. 101 of 1959.\nAppeal by special lea...</td>
      <td>IN-Abs</td>
      <td>The appellants who are displaced persons from ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Appeal No. 52 of 1957.\nAppeal from the judgme...</td>
      <td>IN-Abs</td>
      <td>The appellants and the respondents were owners...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Appeals Nos. 45 and 46 of 1959.\nAppeal by spe...</td>
      <td>IN-Abs</td>
      <td>The respondents firm claimed exemption from Sa...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ION: Criminal Appeal 89 of 1961.\nAppeal by sp...</td>
      <td>IN-Abs</td>
      <td>The appellant was tried for murder.\nThe facts...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Civil Appeal No. 50 of 1961.\nAppeal by specia...</td>
      <td>IN-Abs</td>
      <td>S, employed by the appellant as a cross cutter...</td>
    </tr>
  </tbody>
</table>
</div>



To prepare a knowledge base for legal cases, we transform each row into a document record, which comprises `judgement_case, dataset_name, and summary`.

```python
from langchain_core.documents import Document

docs = []
for (i, doc) in legal_case.iterrows():
    doc = Document(
        page_content=doc['judgement'], 
        metadata={
            "judgement_case": i, 
            "dataset_name": doc["dataset_name"],
            "summary": doc["summary"]
        })
    docs.append(doc)
```


```python
docs[:2]
```


    [Document(metadata={'judgement_case': 0, 'dataset_name': 'IN-Abs', 'summary': "The appellants who are displaced persons from West Pakistan, were granted quasi permanent allotment of some lands in village Raikot in 1949.\nOn October 31, 1952, the Assistant Custodian cancelled the allotment of 14 allottees in village Karodian, and also cancelled the allotment of the Appellants in Raikot but allotted lands to them in village Karodian, and allotted the lands of Raikot to other persons.\nThe 14 allottees of village Karodian as well as the appellants applied for review of the orders of cancellation of their allotment...\n"),
     Document(metadata={'judgement_case': 1, 'dataset_name': 'IN-Abs', 'summary': "The appellants and the respondents were owners of adjoining collieries and the suit out of which the present appeal arose was one brought by the respondents for certain reliefs on the allegation that the appellants had encroached upon their coal mines and removed coal from the encroached portion and that they came to know of the said encroachment and removal of coal after they had received the letter dated August 18, 1941, from the Inspector of Mines...\n")]



Next, we organize a vector database by using `VectorDatabaseFactory` of `aucodb` library. This class orchestrates every necessary methods like chunking and indexing to create a local vector database. We select `milvus` as a search engine for this vector database.


```python
from langchain_huggingface import HuggingFaceEmbeddings
from aucodb.vectordb.factory import VectorDatabaseFactory
from aucodb.vectordb.processor import DocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Initialize document processor
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

doc_processor = DocumentProcessor(splitter=text_splitter)

# 3. Initialize vector database factory
db_type = "milvus"  # Supported types: ['chroma', 'faiss', 'milvus', 'pgvector', 'pinecone', 'qdrant', 'weaviate']
vectordb_factory = VectorDatabaseFactory(
    db_type=db_type,
    embedding_model=embedding_model,
    doc_processor=doc_processor
)

# 4. Store documents in the vector database
vectordb_factory.store_documents(docs)
```

Let's test this vector database by a certain query to extract top-5 similar documents.


```python
query = "claimed exemption from Sales Tax under article 286"
top_k = 5
retrieved_docs = vectordb_factory.query(query, top_k)
for (i, doc) in enumerate(retrieved_docs):
    print(f"Document {i}: {doc}")
```

    Document 0: {'text': "This appeal arises out of the payment of value added tax which was not due, because the supplies in question were exempt..."}
    Document 1: {'text': "This appeal concerns the interpretation of sections 103 and 106 of the Income and Corporation Taxes Act 1988 (ICTA) which..."}
    Document 2: {'text': "This appeal concerns the scope of the duty of confidentiality owed by Her Majestys Revenue and Customs (HMRC) in respect of..."}
    Document 3: {'text': "During the period with which this case is concerned, the claimants (whom we shall refer to as Littlewoods) carried on catalogue sales businesses:..."}
    Document 4: {'text': "This appeal concerns the liability for Value Added Tax (VAT) of a company which markets and arranges holiday accommodation through..."}


Write `semantic_search_query` function to extract relevant chunks based on the semantic meaning represented by embedding vectors.


```python
from typing import Any, Dict, List
def semantic_search_query(query: str, top_k: int) -> List[Dict[str, Any]]:
    if vectordb_factory.vectordb.vector_store is None:
        raise ValueError("Vector store not initialized. Store documents first.")

    # Generate embedding for query
    query_vector = vectordb_factory.vectordb.embedding_model.embed_query(query)

    # Perform similarity search
    results = vectordb_factory.vectordb.client.search(
        collection_name=vectordb_factory.vectordb.collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text"],
        search_params={
            "metric_type": vectordb_factory.vectordb.metric_type
        },  # Use consistent metric type
    )[0]
    returned_docs = [(doc.id, doc.distance) for doc in results]
    return returned_docs

results = semantic_search_query(query=query, top_k=5)
results
```

    [(162, 0.7239072918891907),
     (158, 0.7029068470001221),
     (164, 0.6818636059761047),
     (165, 0.680964469909668),
     (144, 0.6737121939659119)]



In another aspect, we need to consider the overlapping percentage of words between the query and the document. This metric serves as an additional score to improve the ability to extract relevant documents, as it can help address the cases where semantic similarity score is usually high for long sentences.


```python
def exact_match_score(query, doc):
    # Convert strings to sets of words (case-insensitive, removing punctuation)
    query_words = set(query.lower().split())
    doc_words = set(doc.lower().split())
    
    # Calculate intersection of words
    common_words = query_words.intersection(doc_words)
    
    # Avoid division by zero
    if len(query_words) == 0 or len(doc_words) == 0:
        return 0.0
        
    # Calculate score: 0.5 * (|V_q ∩ V_d|/|V_q| + |V_q ∩ V_d|/|V_d|)
    score = 0.5 * (len(common_words) / len(query_words) + len(common_words) / len(doc_words))
    
    return score

def exact_match_search_query(query, docs, top_k: int=5):
    # Calculate scores for all documents
    scores = [(id_doc, exact_match_score(query, doc.page_content)) for (id_doc, doc) in enumerate(docs)]
    
    # Sort by score in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return sorted_scores[:min(top_k, len(docs))]

exact_match_search_query(query=query, docs=docs)
```


    [(2, 0.5056577086280056),
     (87, 0.5027027027027027),
     (53, 0.4397771633051399),
     (162, 0.43877504553734065),
     (194, 0.43830906148867316)]



Next, let's create the SearchLegalEngine class that includes the following functionalities:

- `_create_legal_cases_data`: Creates a legal cases dataset, where each document represents a legal record.

- `_initialize_document_processor`: Creates a vector factory, initializes the vector database, and stores the list of documents.

- `exact_match_search_query`: Computes a score based on the exact matching percentage of overlapping words between the query and the document.

- `semantic_search_query`: Retrieves a list of scores based on semantic meaning.

- `query_fusion_score`: Combines the metrics from exact matching and semantic scores.


```python
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Literal
from typing import Any, Dict, List


class SearchLegalEngine:
    def __init__(self, 
            top_k: int=5, 
            temp_data_path: Path = Path("data/test.parquet"),
            db_type: Literal['chroma', 'faiss', 'milvus', 'pgvector', 'pinecone', 'qdrant', 'weaviate'] = "milvus",
            embedding_model: str="BAAI/bge-small-en-v1.5"
        ):
        self.top_k = top_k
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.temp_data_path = temp_data_path
        self.db_type = db_type
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.doc_processor = DocumentProcessor(splitter=self.text_splitter)

    
    def _download_legal_case(self):
        dataset = load_dataset("joelniklaus/legal_case_document_summarization", split="test")
        dataset.to_parquet(self.temp_data_path)
        

    def _create_legal_cases_data(self):
        self._download_legal_case()
        legal_case = pd.read_parquet(self.temp_data_path)

        self.docs = []
        for (i, doc) in legal_case.iterrows():
            doc = Document(
                page_content=doc['judgement'], 
                metadata={
                    "judgement_case": i, 
                    "dataset_name": doc["dataset_name"],
                    "summary": doc["summary"]
                })
            self.docs.append(doc)
        return self.docs

    def _initialize_document_processor(self):
        self.vectordb_factory = VectorDatabaseFactory(
            db_type=self.db_type,
            embedding_model=self.embedding_model,
            doc_processor=self.doc_processor
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
        score = 0.5 * (len(common_words) / len(query_words) + len(common_words) / len(doc_words))
        
        return score

    def exact_match_search_query(self, query, docs):
        
        # Calculate scores for all documents
        scores = [(id_doc, self.exact_match_score(query, doc.page_content)) for (id_doc, doc) in enumerate(docs)]
        
        return scores
 

    def semantic_search_query(self, query: str, top_k: int=None) -> List[Dict[str, Any]]:
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
        returned_docs = sorted([doc for doc in returned_docs], key=lambda x: x[0], reverse=False)
        return returned_docs
    
    def query_fusion_score(self, query: str, top_k: int=None, threshold: float=None, w_semantic: float=0.5):
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
                    "fusion_score":(1-w_semantic)*exac_score + w_semantic*seman_score 
                }
            )
            for ((id_exac, exac_score), (id_seman, seman_score)) 
                in list(zip(exact_match_scores, semantic_scores))
        ]
        sorted_scores = sorted(scores, key=lambda x: x[1]["fusion_score"], reverse=True)[:min(top_k, len(self.docs))]
        sorted_docs = [(self.docs[i], score) for (i, score) in sorted_scores]
        if threshold:
            filter_docs = [doc for (doc, score) in sorted_docs if score['fusion_score'] > threshold]
            return filter_docs
        else:
            return sorted_docs
```

Test `query_fusion_score` method, which combines the exact matching score and semantic score, to retrieve a list of legal cases related to `Sales Tax`.


```python
search_legal_engine = SearchLegalEngine(
    top_k=5, 
    temp_data_path=Path("data/test.parquet"),
    db_type="milvus",
    embedding_model="BAAI/bge-small-en-v1.5"
)

search_legal_engine._initialize_document_processor()
```

    Repo card metadata block was not found. Setting CardData to empty.
    Creating parquet from Arrow format: 100%|██████████| 1/1 [00:00<00:00, 18.59ba/s]



```python
query = "claimed exemption from Sales Tax"
search_legal_engine.query_fusion_score(query, top_k=5, w_semantic=0.7)
```

    [
        (Document(metadata={'judgement_case': 162, 'dataset_name': 'UK-Abs', 'summary': 'This appeal and cross appeal arise out of claims made by...'}),
        {'semantic_score': 0.7400172352790833,
        'exac_score': 0.5009107468123861,
        'fusion_score': 0.668285288739074}),
        (Document(metadata={'judgement_case': 165, 'dataset_name': 'UK-Abs', 'summary': 'Littlewoods overpaid VAT to HMRC between 1973 and 2004...'}),
        {'semantic_score': 0.7034813165664673,
        'exac_score': 0.4009132420091324,
        'fusion_score': 0.6127108941992668}),
        (Document(metadata={'judgement_case': 2, 'dataset_name': 'IN-Abs', 'summary': 'The respondents firm claimed exemption from Sales Tax under...'}),
        {'semantic_score': 0.6283447742462158,
        'exac_score': 0.5035360678925035,
        'fusion_score': 0.5909021623401021}),
        (Document(metadata={'judgement_case': 87, 'dataset_name': 'IN-Abs', 'summary': 'Each of the appellants/petitioners is a registered dealer in...'}),
        {'semantic_score': 0.6285038590431213,
        'exac_score': 0.5016891891891891,
        'fusion_score': 0.5904594580869417}),
        (Document(metadata={'judgement_case': 154, 'dataset_name': 'UK-Abs', 'summary': 'The benefit cap was introduced in the Welfare Reform Act 2012...'}),
        {'semantic_score': 0.613065779209137,
        'exac_score': 0.5004644250417982,
        'fusion_score': 0.5792853729589353})
    ]

Now, let's save code into a module file `vinagent/tools/legal_assistant/search_legal_cases.py`, which will be loaded as a search tool in the next case.

```python
%%writefile vinagent/tools/legal_assistant/search_legal_cases.py
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
    def __init__(self, 
            top_k: int=5, 
            temp_data_path: Path = Path("data/test.parquet"),
            db_type: Literal['chroma', 'faiss', 'milvus', 'pgvector', 'pinecone', 'qdrant', 'weaviate'] = "milvus",
            embedding_model: str="BAAI/bge-small-en-v1.5"
        ):
        self.top_k = top_k
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.temp_data_path = temp_data_path
        self.db_type = db_type
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.doc_processor = DocumentProcessor(splitter=self.text_splitter)

    
    def _download_legal_case(self):
        dataset = load_dataset("joelniklaus/legal_case_document_summarization", split="test")
        dataset.to_parquet(self.temp_data_path)
        

    def _create_legal_cases_data(self):
        self._download_legal_case()
        legal_case = pd.read_parquet(self.temp_data_path)

        self.docs = []
        for (i, doc) in legal_case.iterrows():
            doc = Document(
                page_content=doc['judgement'], 
                metadata={
                    "judgement_case": i, 
                    "dataset_name": doc["dataset_name"],
                    "summary": doc["summary"]
                })
            self.docs.append(doc)
        return self.docs


    def _initialize_document_processor(self):
        self.vectordb_factory = VectorDatabaseFactory(
            db_type=self.db_type,
            embedding_model=self.embedding_model,
            doc_processor=self.doc_processor
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
        score = 0.5 * (len(common_words) / len(query_words) + len(common_words) / len(doc_words))
        
        return score

    def exact_match_search_query(self, query, docs):
        # Calculate scores for all documents
        scores = [(id_doc, self.exact_match_score(query, doc.page_content)) for (id_doc, doc) in enumerate(docs)]
        return scores
 

    def semantic_search_query(self, query: str, top_k: int=None) -> List[Dict[str, Any]]:
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
        returned_docs = sorted([doc for doc in returned_docs], key=lambda x: x[0], reverse=False)
        return returned_docs

    def query_fusion_score(self, query: str, top_k: int=None, threshold: float=None, w_semantic: float=0.5):
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
                    "fusion_score":(1-w_semantic)*exac_score + w_semantic*seman_score 
                }
            )
            for ((id_exac, exac_score), (id_seman, seman_score)) 
                in list(zip(exact_match_scores, semantic_scores))
        ]
        sorted_scores = sorted(scores, key=lambda x: x[1]["fusion_score"], reverse=True)[:min(top_k, len(self.docs))]
        sorted_docs = [(self.docs[i], score) for (i, score) in sorted_scores]
        if threshold:
            filter_docs = [doc for (doc, score) in sorted_docs if score['fusion_score'] > threshold]
            return filter_docs
        else:
            return sorted_docs

@primary_function
def query_similar_legal_cases(query: str, n_legal_cases: int=2, threshold: float=0.6):
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
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    search_legal_engine._create_legal_cases_data()
    search_legal_engine._initialize_document_processor()
    docs = search_legal_engine.query_fusion_score(query, top_k=n_legal_cases, threshold=threshold, w_semantic=0.7)
    return docs
```

    Overwriting vinagent/tools/legal_assistant/search_legal_cases.py


## Initialize Legal Agent

We demonstrate how to initialize a legal assistant on vinagent, which can assist users with tasks such as:

- Searching for relevant legal cases.

- Summarizing the major timeline of events in a specific legal case.

- Analyzing arguments to identify the strengths and weaknesses of the appellants’ claims.

- Conducting jurisdictional analysis of the Act and Regulation.

- Analyzing ethics and potential bias in court rulings.

Let's initialize the legal_agent with relevant description, skills, and tools.

```python
from vinagent.agent.agent import Agent
from langchain_openai import ChatOpenAI
from vinagent.agent.agent import Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

llm = ChatOpenAI(
    model = "o4-mini"
)

legal_agent = Agent(
    name="Legal Assistant",
    description="A legal assistant who can find the similar legal cases",
    llm = llm,
    skills=[
        "search similar legal cases",
        "summary the legal cases",
        "extract the main timeline in the legal cases",
        "search information on the internet"
    ],
    tools=[
        'vinagent/tools/legal_assistant/search_legal_cases.py',
        'vinagent/tools/websearch_tools.py'
    ]
)
```

    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered query_similar_legal_cases:
    {'tool_name': 'query_similar_legal_cases', 'arguments': {'n_legal_cases': 2, 'threshold': 0.6}, 'return': 'The similar legal cases.', 'docstring': 'Query the similar legal cases to the given query.\n    Args:\n        query (str): The query string.\n        n_legal_cases (int): The number of legal cases\n        threshold (float): The similarity threshold. Defaults to 0.6.\n    Returns:\n        The similar legal cases.', 'dependencies': ['pandas', 'datasets', 'pathlib', 'langchain_core', 'langchain.text_splitter', 'langchain_huggingface', 'aucodb', 'vinagent'], 'module_path': 'vinagent.tools.search_legal_cases', 'tool_type': 'module', 'tool_call_id': 'tool_331a2f3a-8bc9-454a-89b3-2ad3c8ac074e'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.search_legal_cases
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered search_api:
    {'tool_name': 'search_api', 'arguments': 'query: Union[str, dict[str, str]]', 'return': 'The answer from search query', 'docstring': 'Search for an answer from a query string\nArgs:\n    query (dict[str, str]):  The input query to search\nReturns:\n    The answer from search query', 'dependencies': ['os', 'dotenv', 'tavily', 'dataclasses', 'typing', 'vinagent.register'], 'module_path': 'vinagent.tools.websearch_tools', 'tool_type': 'module', 'tool_call_id': 'tool_f0647aec-0361-4791-9f05-ffa05f2d9f98'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.websearch_tools


## Find the relevant legal case

Attorney usually finds the relevant legal cases to prepare before starting a lawsuit. The primary target of finding similar legal cases is to identify relevant precedents that guide the resolution of a current case, ensuring consistency and fairness in legal outcomes. By researching cases with comparable facts or legal issues, attorneys can build stronger arguments, predict judicial rulings, and uncover defenses or counterarguments. This process supports compliance with the principle of stare decisis, enhances case strategy, and provides leverage in negotiations, ultimately saving time and resources while grounding legal decisions in established judicial authority.


```python
message = legal_agent.invoke(
    "Let find one legal case claimed exemption sales tax", 
    is_tool_formatted=False,
    max_history=1
)
message
```

    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'legal case sales tax exemption claimed'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'legal case sales tax exemption claimed'})

    ToolMessage(content="Completed executing module tool search_api({'query': 'legal case sales tax exemption claimed'})", tool_call_id='tool_f0647aec-0361-4791-9f05-ffa05f2d9f98', artifact="Sales tax exemptions were claimed in various legal cases, including a Mississippi case about medical devices and a Missouri case about electronic price scanners. The Supreme Court struck down a Texas statute exempting religious publications from sales tax. Nonprofits' tax exemptions were also challenged.")


This is the main content of a similar legal case.


```python
message.artifact
```

    "Sales tax exemptions were claimed in various legal cases, including a Mississippi case about medical devices and a Missouri case about electronic price scanners. The Supreme Court struck down a Texas statute exempting religious publications from sales tax. Nonprofits' tax exemptions were also challenged."



In there, only use `is_tool_formatted=False` while disabling modification tool message in the next step. We set `max_history=1` to enable only the last query in current context and remove history. That ensures the context length does not exceed the maximum length accepted by the LLM, which usually happens while legal cases have long context.


```python
legal_agent.in_conversation_history.get_history()
```

    [SystemMessage(content='A legal assistant who can find the similar legal cases\nHere is your skills:\n- search similar legal cases- summary the legal cases- extract the main entities in the legal cases- search information on the internet', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='You are given a task, a list of available tools, and the memory about user to have precise information.\n- Task: Let find one legal case claimed exemption sales tax\n- Tools list: {"search_api": {"tool_name": "search_api", "arguments": "query: Union[str, dict[str, str]]", "return": "The answer from search query", "docstring": "Search for an answer from a query string\\nArgs:\\n    query (dict[str, str]):  The input query to search\\nReturns:\\n    The answer from search query", "dependencies": ["os", "dotenv", "tavily", "dataclasses", "typing", "vinagent.register"], "module_path": "vinagent.tools.websearch_tools", "tool_type": "module", "tool_call_id": "tool_f0647aec-0361-4791-9f05-ffa05f2d9f98"}, "trending_news_google_tools": {"tool_name": "trending_news_google_tools", "arguments": {"top_k": 5, "topic": "AI", "host_language": "en-US", "geo_location": "US"}, "return": "a list of dictionaries containing the title, link, and summary of the top trending news", "docstring": "Summarize the top trending news from Google News from a given topic.", "dependencies": ["requests", "BeautifulSoup", "pandas", "langchain_together", "googlenewsdecoder", "dotenv"], "module_path": "vinagent.tools.trending_news", "tool_type": "module", "tool_call_id": "tool_56da458b-7ac5-49da-ac2e-2b7857a69817"}, "deepsearch_tool": {"tool_name": "deepsearch_tool", "arguments": {"query": "str", "max_chapters": "4", "max_paragraphs_per_chapter": "5", "max_critical_queries": "3", "max_revisions": "1"}, "return": "str", "docstring": "Invoke deepsearch to deeply analyze the query and generate a more detailed response.", "dependencies": ["os", "re", "typing", "pydantic", "dotenv", "langgraph", "langchain_core", "langchain_together", "tavily"], "module_path": "vinagent.tools.deepsearch", "tool_type": "module", "tool_call_id": "tool_f5568b04-72fe-4885-8494-3f8c121729ba"}, "query_similar_legal_cases": {"tool_name": "query_similar_legal_cases", "arguments": {"n_legal_cases": 2, "threshold": 0.6}, "return": "The similar legal cases.", "docstring": "Query the similar legal cases to the given query.\\n    Args:\\n        query (str): The query string.\\n        n_legal_cases (int): The number of legal cases\\n        threshold (float): The similarity threshold. Defaults to 0.6.\\n    Returns:\\n        The similar legal cases.", "dependencies": ["pandas", "datasets", "pathlib", "langchain_core", "langchain.text_splitter", "langchain_huggingface", "aucodb", "vinagent"], "module_path": "vinagent.tools.search_legal_cases", "tool_type": "module", "tool_call_id": "tool_331a2f3a-8bc9-454a-89b3-2ad3c8ac074e"}}\n\n- User: unknown_user\n------------------------\nInstructions:\n- Let\'s answer in a natural, clear, and detailed way without providing reasoning or explanation.\n- If user used I in Memory, let\'s replace by name unknown_user in User part.\n- You need to think about whether the question need to use Tools?\n- If it was daily normal conversation. Let\'s directly answer as a human with memory.\n- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n{\n"tool_name": "Function name",\n"tool_type": "Type of tool. Only get one of three values ["function", "module", "mcp"]"\n"arguments": "A dictionary of keyword-arguments to execute tool_name",\n"module_path": "Path to import the tool"\n}\n- Let\'s say I don\'t know and suggest where to search if you are unsure the answer.\n- Not make up anything.\n', additional_kwargs={}, response_metadata={}),
     AIMessage(content='{"tool_name": "search_api", "tool_type": "module", "arguments": {"query": "legal case sales tax exemption claimed"}, "module_path": "vinagent.tools.websearch_tools"}', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 513, 'prompt_tokens': 1000, 'total_tokens': 1513, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 448, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'o4-mini-2025-04-16', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--a3551d66-37ba-4685-91b1-9954cb8e917e-0', usage_metadata={'input_tokens': 1000, 'output_tokens': 513, 'total_tokens': 1513, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 448}}),
     ToolMessage(content="Completed executing module tool search_api({'query': 'legal case sales tax exemption claimed'})", tool_call_id='tool_f0647aec-0361-4791-9f05-ffa05f2d9f98', artifact="Sales tax exemptions were claimed in various legal cases, including a Mississippi case about medical devices and a Missouri case about electronic price scanners. The Supreme Court struck down a Texas statute exempting religious publications from sales tax. Nonprofits' tax exemptions were also challenged.")]



The history only returns a list of messages ending with ToolMessage. If you want the Agent to modify the ToolMessage according to human preferences, set `is_tool_formatted=True`.


```python
message = legal_agent.invoke(
    "Let find one legal case claimed exemption sales tax", 
    is_tool_formatted=True,
    max_history=1
)
message
```

    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'legal case claimed exemption sales tax'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'legal case claimed exemption sales tax'})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10

    AIMessage(content='One prominent example is Texas Monthly, Inc. v. Bullock, 489 U.S. 1 (1989).  In that case Texas Monthly challenged a state statute that exempted from sales tax only those periodicals devoted exclusively to religious or public affairs.  The U.S. Supreme Court held the exemption unconstitutional under the Establishment Clause.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 536, 'prompt_tokens': 1061, 'total_tokens': 1597, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 448, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'o4-mini-2025-04-16', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--a501aa03-3411-4b44-9a11-440bb9c25273-0', usage_metadata={'input_tokens': 1061, 'output_tokens': 536, 'total_tokens': 1597, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 448}})




```python
legal_agent.in_conversation_history.get_history()
```


    [SystemMessage(content='A legal assistant who can find the similar legal cases\nHere is your skills:\n- search similar legal cases- summary the legal cases- extract the main entities in the legal cases- search information on the internet', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='You are given a task, ...'),
     AIMessage(content='Here are some of the main ethical considerations ...'),
     SystemMessage(content='A legal assistant who can find the similar legal cases ...'),
     HumanMessage(content='You are given a task, a list of available tools ...'),
     AIMessage(content='{"tool_name": "search_api", "tool_type": "module", "arguments": {"query": "legal case claimed exemption sales tax"}, "module_path": "vinagent.tools.websearch_tools"}'),
     ToolMessage(content="Completed executing module tool search_api()..."),
     SystemMessage(content='A legal assistant who can find the similar legal cases\nHere is your skills:\n- search similar legal cases- summary the legal cases- extract the main entities in the legal cases- search information on the internet'),
     HumanMessage(content='You are given a task, a list of available tools...'),
     AIMessage(content='One prominent example is Texas Monthly, Inc. v. Bullock, 489 U.S. 1 (1989).  In that case Texas Monthly challenged a state statute that exempted from sales tax only those periodicals devoted exclusively to religious or public affairs.  The U.S. Supreme Court held the exemption unconstitutional under the Establishment Clause.')]


By default, VinAgent’s agent can store up to the last 10 messages in its conversation history. Therefore, if we continue the query, the list of answers will be appended to the existing history and the most old message will be popped out according to FIFO (first in first out). In this case, if you choose to modify the tool result, you will receive an `AIMessage` as the last message.


## Summarize legal case

With very long legal cases, we cannot capture each event in detail. Therefore, we need to summarize the legal case in a short form to accelerate attorneys’ reading speed. Let's test summary with legal case 199th.

```python
legal_case = docs[199].page_content
message = legal_agent.invoke(f"Let's summarize this legal case in 200 words including context, development, plaintiff's arguments, and court ruling \n{legal_case}", max_history=1)
```

    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 1 iterations.



```python
print(message.content)
```

    This case centers on the legal framework governing a Development Consent Order (DCO) for Heathrow Airport Ltd’s (HAL) proposal to build a third runway (the North-West Runway, NWR). Successive UK governments and the independent 2012 Airports Commission concluded that additional airport capacity in southeast England was needed by 2030. After extensive public and expert consultation, the Government issued a draft National Policy Statement (NPS) under the Planning Act 2008, culminating in the Airports NPS (ANPS) in June 2018. Objectors—Friends of the Earth and Plan B Earth—claimed the Secretary of State unlawfully ignored the Paris Agreement’s temperature targets, failed to explain how the ANPS aligned with Government climate policy (section 5(8)), lacked proper regard for climate change mitigation (section 10), and breached the Strategic Environmental Assessment Directive by omitting non-CO₂ impacts and post-2050 emissions. The Divisional Court refused permission, finding the Secretary of State had rationally relied on the Climate Change Act 2008 and expert advice from the Committee on Climate Change. The Court of Appeal overturned that decision, declaring the ANPS legally ineffective. HAL then appealed to the Supreme Court, which allowed HAL’s appeal. The Supreme Court held that neither the Paris Agreement nor unquantified non-CO₂ or post-2050 emissions were “obviously material” legal requirements at the policy-setting stage and that the ANPS had lawfully discharged statutory duties.


## Timeline and Fact Organization

We can organize the timeline of events for each legal case in reverse chronological order to enhance tracking efficiency. This allows an attorney to review the event sequence based on the timeline to identify key developments in the lawsuit. This approach is especially valuable for lengthy and complex legal cases, where recalling every minor event is essential.


```python
message = legal_agent.invoke(f"Let's create a timeline of events in this legal case in descending order: \n{legal_case}", max_history=1)
```

    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 1 iterations.



```python
print(message.content)
```

    Here is a descending‐order timeline of the key events in the Heathrow 3rd-runway/NPS litigation and its policy background:
    
    • 25 June 2020  
      – Committee on Climate Change publishes its 2020 Progress Report recommending inclusion of international aviation and shipping in the UK’s net-zero targets and urging review of airport capacity strategy in light of COVID-19 and net zero.
    
    • 26 June 2019  
      – The Climate Change Act 2008 (2050 Target Amendment) Order 2019 comes into force, amending section 1 of the CCA 2008 to require a 100% reduction in net UK carbon emissions by 2050 (i.e. “net zero”).
    
    • 25 June 2019  
      – Parliament declares a “climate and environmental emergency.”  
      – On 26 June, the CCC’s report “Net Zero: The UK’s contribution to stopping global warming” is published, advising urgent statutory adoption of net-zero greenhouse gas targets by 2050 (including international aviation/shipping).
    
    • 24 September 2019  
      – CCC writes to the Secretary of State for Transport recommending formal inclusion of international aviation and shipping in the UK’s net-zero statutory targets.
    
    • May 2019 ([2019] EWHC 1070 (Admin); [2019] EWHC 1069 (Admin))  
      – The Divisional Court (Hickinbottom LJ & Holgate J) hears the rolled-up claims and dismisses all challenges to the designation of the Airports NPS (ANPS).
    
    • 26 February 2020 ([2020] EWCA Civ 214)  
      – Court of Appeal allows the Friends of the Earth and Plan B Earth appeals, declares the ANPS unlawful and of no legal effect for failure to take the Paris Agreement properly into account.
    
    • February 2017  
      – Department for Transport launches first public consultation on a draft Airports National Policy Statement (ANPS) for the northwest-runway (NWR) scheme.
    
    • October 2017  
      – Second round of consultation on the draft ANPS is opened (large numbers of responses received).
    
    • 1 November 2017  
      – Transport Committee publishes its report on the proposed NWR scheme; Government later issues a formal response in June 2018.
    
    • June 2018  
      – DfT publishes its responses to both rounds of ANPS consultation and to the Transport Committee report.
    
    • 5 June 2018  
      – Secretary of State lays the final version of the ANPS before Parliament, together with its Sustainability Appraisal (the SEA/AA reports).
    
    • 25 June 2018  
      – House of Commons debates and votes to approve the ANPS (415 to 119).
    
    • 26 June 2018  
      – Secretary of State formally designates the Airports NPS under section 5(1) of the Planning Act 2008.
    
    • Late June–July 2018  
      – Objectors (including Friends of the Earth and Plan B) issue claims for judicial review under section 13 of the Planning Act 2008, challenging the lawfulness of the ANPS designation.
    
    • 14 June 2018  
      – CCC Chair (Lord Deben) and Deputy Chair (Baroness Brown) write to the Transport Secretary expressing surprise that the ANPS did not refer to the CCA 2008 targets or the Paris Agreement.
    
    • 5 June 2018 (same day as laying the ANPS)  
      – Cabinet sub-committee paper confirms the government will address aviation emissions in its forthcoming Aviation Strategy; notes current uncertainty over carbon-budget treatment of international aviation.
    
    • 17 April 2018  
      – At the Commonwealth Heads of Government Meeting, UK announces it will seek CCC advice on Paris Agreement implications once the IPCC’s 1.5 °C report is published.
    
    • December 2017  
      – DfT publishes “Beyond the Horizon: The Future of UK Aviation – Next Steps,” setting out how the forthcoming Aviation Strategy will address both CO₂ and non-CO₂ aviation climate impacts.
    
    • October 2016  
      – CCC publishes advice (“UK Climate Action following the Paris Agreement”) confirming that existing UK targets remain appropriate for now and advising they be kept under review.
    
    • 14 March & 24 March 2016  
      – Ministers in the House of Commons state that the UK intends to enshrine a “net-zero” Paris goal in domestic law but that the questions of “how” remain to be answered.
    
    • 22 April 2016  
      – United Kingdom signs the Paris Agreement.
    
    • 17 November 2016  
      – United Kingdom ratifies the Paris Agreement.
    
    • 25 October 2016  
      – Transport Secretary announces the north-west-runway (NWR) option as the government’s preferred scheme for Heathrow expansion.
    
    • 12 December 2015  
      – Paris Agreement text is agreed at COP 21.
    
    • 14 December 2015  
      – Transport Secretary announces the government will proceed via a national policy statement (NPS) under the Planning Act 2008 and that further work on environmental impacts (including carbon) is required.
    
    • 1 July 2015  
      – Airports Commission publishes its Final Report, selecting the northwest-runway (NWR) option as its “preferred” solution (subject to mitigation).
    
    • 17 December 2013  
      – Airports Commission publishes its Interim Report, concluding that one new runway is needed by 2030 and assessing options under a carbon-trading cap consistent with a 2 °C goal.
    
    • 2012  
      – Airports Commission is established under Sir Howard Davies to review UK airport capacity and recommend a scheme.
    
    • 26 November 2008 (approx.)  
      – Climate Change Act 2008 and Planning Act 2008 are enacted on the same day, creating the framework for UK carbon targets, carbon budgets and a new nationally significant infrastructure consenting regime (via NPSs and DCOs).
    
    • 1992  
      – United Nations Framework Convention on Climate Change is adopted, laying the foundation for later climate treaties (including Kyoto and Paris).


## Argument analysis

Sometimes, attorney needs dive deepth to understand the strength and weakness of appellants' arguments. This is to ensure they can increase the probability of win before the trial begins. legal_agent can also deeply analyze the strength and weakness. This process involves evaluating evidence, legal precedents, and potential counterarguments to build a robust strategy.


```python
message = legal_agent.invoke(f"Let's analyze the strengths and weaknesses of appellants' arguments: \n{legal_case}", max_history=1)
```

    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 1 iterations.



```python
print(message.content)
```

    Below is a concise appraisal of the four principal grounds on which Heathrow Airport Ltd (HAL) (together with the Secretary of State) challenged the Court of Appeal’s decision invalidating the Airports National Policy Statement (ANPS).  For each ground I identify what I see as the chief strengths of HAL’s position and the major vulnerabilities in FoE/Plan B’s counter-arguments.
    
    1. Ground (i): Section 5(8) PA 2008 – “Duty to explain how the NPS takes account of Government climate policy”  
       HAL’s argument  
       • Parliament fixed the UK’s carbon-reduction commitment in the Climate Change Act 2008.  That target (80 % by 2050, later 100 %) and the process for adjusting it are entrenched in statute.  The ANPS properly explained how it took that settled policy into account (paras 5.71–5.73).  Anything beyond that—e.g. aspirational global temperature goals in the Paris Agreement—was not “Government policy” in the sense of an established, unqualified domestic statement of policy.  
       Strengths  
       – Textual: section 5(8) looks to “Government policy” already laid down.  Treaties in force but not given domestic effect—and ministerial utterances that flag only an intention to return to Parliament later—do not qualify.  
       – Context and purpose: NPS explanations must avoid endless trawls through speeches, consultation responses or emerging international commitments.  They must stick to clear, binding domestic policy statements.  
       Weaknesses for FoE/Plan B  
       – They relied on loose ministerial remarks in early 2016 and on the Paris Agreement’s temperature goal.  Those neither created binding UK law nor amounted to an unqualified Government policy of the kind that section 5(8) demands.  
       – Invoking aspirational treaty aims would impose a burden on NPS drafters to catalogue every “policy” mention in White Papers, Commons Statements, press releases, etc., leading to unpredictability.  
    
    2. Ground (ii): Section 10 PA 2008 – “Have regard to mitigation/adaptation”  
       HAL’s argument  
       • Section 10 requires the Secretary of State to aim for sustainable development, including mitigating/adapting to climate change, but gives a broad margin of judgment as to which material considerations to weigh and the weight to give them.  The ANPS addressed the UK’s binding obligations under the Climate Change Act, which already give effect to the Paris Agreement NDCs.  The Paris Agreement itself is not extra domestic law and needed no separate treatment.  
       Strengths  
       – Well-settled Wednesbury principle: dehors express statutory direction, a decision-maker need not catalogue every possible international commitment.  Unless omission is irrational, it is lawful.  
       – The ANPS did address greenhouse gases, carbon budgets, and required that any DCO application demonstrate consistency with whatever targets applied at that later date.  
       Weaknesses for FoE/Plan B  
       – Their approach conflated international treaty goals with domestic law obligations.  They could not show that the Secretary of State’s refusal even arguable lacked “reasonableness” or was outside his wide discretion.  
       – To impose a free-standing duty to re-examine global aims would overstep parliament’s careful design of carbon-budget review processes in the 2008 Act.  
    
    3. Ground (iii): SEA Directive – “Adequacy of the Environmental Report”  
       HAL’s argument  
       • The strategic environmental assessment (SEA) for the draft ANPS took the Airports Commission’s extensive work (including all carbon scenarios) as its baseline.  It quantified CO₂ and recorded that non-CO₂ effects were uncertain and could not yet be sensibly modelled.  Under Articles 5(2)–(3) SEA (and transposing regs), the Secretary of State had a broad discretion to decide what “reasonably may be required” in an environmental report.  The public was consulted and responses on climate matters were taken into account.  
       Strengths  
       – Blewett/Wednesbury standard: environmental reports need only supply a sufficient basis for public consultation, not endless academic coverage of every hypothetical.  
       – The public consultation process did solicit comments on climate goals; the report and ANPS responded.  
       Weaknesses for FoE/Plan B  
       – Their claim rested on a judicial curiosity—that the Paris Agreement should have been named in the scoping list.  But law and policy on aviation emissions were evolving; the report explained clearly why non-CO₂ effects were deferred for future, more detailed work.  
       – Under EU law as under domestic law, the SEA Directive tolerates an iterative and proportionate approach.  
    
    4. Ground (iv): Section 10(3) PA 2008 – “Post-2050 and Non-CO₂ Emissions”  
       HAL’s argument  
       • The ANPS (and its AoS) modelled airport emissions through 2085/86 and quantified all CO₂ emissions (pre- and post-2050).  They explained that post-2050 policy settings and future carbon budgets would govern any DCO, and that non-CO₂ climate impacts are scientifically uncertain and not yet regulated.  Those matters will be addressed in the forthcoming Aviation Strategy and at the DCO stage.  It was neither irrational nor procedurally defective to defer them.  
       Strengths  
       – The ANPS spelled out the requirement that any DCO applicant show consistency with “then current” targets.  There is a built-in safety valve (section 104 PA 2008) permitting refusal if future obligations would be breached.  
       – Reasonable scientific uncertainty about non-CO₂ forcings, and absence of an agreed metric, justified a focused CO₂ assessment now and detailed post-2030 work later.  
       Weaknesses for FoE/Plan B  
       – Invoking a notional “net zero” objective for post-2050 emissions would have required guesses about legislation not yet in place.  That cannot render a policy statement unlawful.  
       – Their critique collapses into an argument for pre-deciding a DCO refusal point far in advance of any application, blurring the line between NPS-making and DCO decision-making.  
    
    Overall conclusion  
       HAL’s key legal strength across all grounds is that Parliament deliberately embedded UK carbon commitments in the Climate Change Act framework, prescribing a process for adjusting them.  Neither the Paris Agreement’s aspirational temperature goals nor emerging Aviation Strategy work constituted binding domestic policy or mandatory inputs at the NPS-making stage.  The courts below (Divisional Court, then Sup Ct on appeal) correctly recognized the broad discretion and margin of judgment that the PA 2008 and SEA regime afford to decision-makers on these complex, evolving climate issues.


## Jurisdictional Analysis

Jurisdictional analysis is vital in legal proceedings to ensure challenges are pursued correctly and efficiently. It serves to identify the correct legal framework, ensure compliance with time limits, define the scope of review, clarify court hierarchy and appeal routes, guide remedies and outcomes, and align with statutory interplay. By addressing these aspects, jurisdictional analysis prevents procedural errors, focuses arguments on permissible legal grounds, and informs strategic decisions, thereby upholding the integrity of the judicial process.


```python
message = legal_agent.invoke(f"Let's analyze the jurisdictional analysis: \n{legal_case}", max_history=1)
print(message.content)
```

    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    The challenge to the Secretary of State’s designation of the Airports National Policy Statement (ANPS) went through the courts not by way of ordinary appeal but by judicial review under the Planning Act 2008 and the Senior Courts Act 1981.  The jurisdictional framework can be broken down as follows:
    
    1.   Statutory basis for challenge (PA 2008 s 13)  
         –  Section 13(1) of the Planning Act 2008 provides the *only* route by which an NPS may be questioned in the courts: by way of a claim for judicial review.  
         –  It must be brought within six weeks from the later of (a) the date of designation of the NPS or (b) its publication.  
    
    2.   Procedural history  
         –  Objectors (Friends of the Earth, Plan B Earth and others) filed judicial review proceedings in the High Court challenging the Secretary of State’s decision under s 5(1) to designate the ANPS.  
         –  The Divisional Court (Hickinbottom LJ and Holgate J) heard all of the claims on a “rolled-up” basis (i.e. permission and merits together) and dismissed every ground.  
         –  The objectors appealed under the statutory right of appeal from a JR decision to the Court of Appeal.  That court allowed two of the grounds, quashed the designation, and said the ANPS was of no legal effect unless and until remedial steps were taken.  
         –  Heathrow Airport Ltd (HAL) (joined as an interested party below) then obtained permission to appeal to the Supreme Court under s 40(1) of the Constitutional Reform Act 2005.  
    
    3.   Scope of judicial review jurisdiction  
         –  Subject-matter jurisdiction: only the lawfulness of the NPS designation itself (and the procedures leading up to it) could be reviewed, not the merits of airport expansion or technical climate policy.  
         –  Time limit: six weeks under PA 2008 s 13(1).  
         –  Remedies: the court can quash the designation or grant declarations (but not award damages).  The Secretary of State could then (and still can) choose to re-designate the ANPS after remedying any procedural or legal defects.  
    
    4.   Relationship to other statutory provisions  
         –  Senior Courts Act 1981 s 31 gives the High Court (and Divisional Court) general JR jurisdiction; PA 2008 s 13 is a carve-out stipulating the subject and time limit for NPS challenges.  
         –  No ordinary “appeal” lies against an NPS under PA 2008 s 9—only JR under s 13.  
         –  Once an NPS is in force, applications for Development Consent Orders under PA 2008 Part 6 must be determined “in accordance with” the NPS (s 104), subject to limited exceptions.  
    
    5.   Final leave and outcome  
         –  HAL, as an interested party with a substantial investment in the NWR proposal, was granted leave to defend the validity of the ANPS in the Supreme Court.  
         –  The Supreme Court’s ultimate task is purely to resolve the legal questions of statutory interpretation and public-law review raised by the appeal.  
    
    In short, the courts’ jurisdiction in this case sprang from the Planning Act’s carefully circumscribed JR regime for national policy statements (PA 2008 s 13), carried through the Divisional Court, Court of Appeal and now the Supreme Court, with all stakeholders bound by the six-week time bar and limited to questions of law and procedure.


## Ethical and Bias in court ruling

Court rulings need to be ethical and free from bias to deliver fair, open, and responsible decisions, especially in tricky cases while many unfair judgements were made. It’s about ensuring justice for future generations, weighing economic gains against environmental and local community impacts, being transparent, handling scientific unknowns carefully, avoiding institutional blind spots, and striking the right balance in judicial oversight. If courts don’t tackle these ethical and bias issues head-on, they risk deepening inequalities, weakening environmental protections, and losing the public’s trust in the system.


```python
message = legal_agent.invoke(f"Let's consider the ethical and bias arguments of court ruling for this legal case: \n{legal_case}", max_history=1)
print(message.content)
```

    INFO:vinagent.agent.agent:Tool calling iteration 1/10


    Here are some of the main ethical considerations and potential bias-related issues raised by the courts’ treatment of the third-runway policy statement (ANPS) for Heathrow:  
    
    1. Intergenerational and global justice  
    •   Duty to future generations:  Expanding Heathrow locks in additional carbon emissions for decades. Ethically, do today’s beneficiaries have a right to harness fossil-fuel aviation capacity at the expense of young people and unborn generations bearing the climate cost?  
    •   “Common but differentiated responsibility”:  The UK has among the stricter legally binding climate targets. There is a moral question whether it ought to go beyond its 2050 carbon ceiling (and take non-CO₂ effects into account) so as to set a leadership example globally.  
    
    2. Weighing economic gains vs. environmental harms  
    •   Distributional impact on local communities:  The project will bring noise, air pollution and property blight to residents under flight paths. An ethical analysis probes whether their health and quality-of-life costs have been fairly balanced against national GDP growth and passenger convenience.  
    •   Procedural fairness:  Did the consultation and environmental assessment processes genuinely empower affected communities to shape or challenge the policy, or were they a box-ticking exercise that privileged central government’s economic case?  
    
    3. Transparency and reason-giving  
    •   Climate rationale omitted:  The Court of Appeal found the Secretary of State failed to explain how the ANPS policy aligned with the Paris goals; the Supreme Court later said that omission did not cross the threshold of irrationality. Ethically, stakeholders argue the public deserves an explicit account of how airport expansion sits alongside the U.K.’s broader commitments to limit warming to “well below 2 °C.”  
    •   Trust in expert advice:  Relying heavily on the Airports Commission and the Committee on Climate Change can seem to bias decision-making toward technical or economic expertise while sidelining lay and community values around environmental precaution.  
    
    4. Use of the precautionary principle vs. innovation optimism  
    •   Precautionary stance:  Some ethicists argue that “scientific uncertainty” over non-CO₂ impacts and post-2050 emissions should have triggered a precautionary moratorium until tighter metrics and policies were in place.  
    •   Innovation and efficiency view:  Government and HAL contend that technology improvements (more fuel-efficient aircraft, carbon trading, sustainable aviation fuels) will allow capacity growth without breaching climate targets—an ethically forward-looking, solution-driven stance.  
    
    5. Institutional and cognitive biases  
    •   Status-quo momentum (“path-dependency”):  Heathrow has long been a focal point of U.K. aviation policy. Ethically, decision-makers may be unduly anchored to existing infrastructure and past studies rather than re-evaluating fundamental climate and social trade-offs.  
    •   Framing and risk perception:  By characterizing carbon emissions as “not a reason to refuse consent unless material,” the ANPS frames climate risk as secondary, which can bias assessments against deeply weighting environmental uncertainties.  
    
    6. Judicial deference vs. rights-protecting oversight  
    •   Deference to political branches:  The Supreme Court majority accepted that the Secretary of State’s policy judgments—even if contestable—were not irrational. Some critics see this as ethically problematic deference that weakens environmental accountability.  
    •   Activist impulse:  Conversely, the Court of Appeal’s willingness to quash the ANPS for failure to expressly link to Paris raised questions about the proper reach of judicial review and the democratic mandate of Parliamentarians who voted in support.  
    
    In sum, this ruling sits at the intersection of competing ethical claims—economic growth and social opportunity versus climate justice and community well-being—and highlights how technical advice, statutory formulations and institutional roles can introduce biases in how those claims are weighed.
