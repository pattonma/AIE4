from datasets import Dataset
from langchain_core.runnables import RunnableSequence
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
import constants
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
import tqdm

def generateDataset(chain: RunnableSequence, test_questions: list, test_groundtruths: list) -> Dataset:
    answers = []
    contexts = []
    for question in test_questions:
        response = chain.invoke({"question" : question})
        answers.append(response["response"].content)
        contexts.append([context.page_content for context in response["context"]])
    
    response_dataset = Dataset.from_dict({
        "question" : test_questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : test_groundtruths
    })
    return response_dataset

def generateQdrantRetriever(documents: list, embeddingModel: Embeddings, nameExt='') -> BaseRetriever:
    qdrant_client = QdrantClient(constants.LOCATION)

    qdrant_client.create_collection(
        collection_name=constants.COLLECTION_NAME+nameExt,
        vectors_config=VectorParams(size=constants.VECTOR_SIZE, distance=Distance.COSINE),
    )

    qdrant_vs = QdrantVectorStore(
        client=qdrant_client,
        collection_name=constants.COLLECTION_NAME+nameExt,
        embedding=embeddingModel,
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]

    qdrant_vs.add_documents(documents=documents, ids=uuids)

    retriever = qdrant_vs.as_retriever()

    return retriever

def generateChain(retriever: BaseRetriever) -> RunnableSequence:
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": constants.BASE_RAG_PROMPT | constants.BASE_LLM, "context": itemgetter("context")}
    )
    return chain

def create_questions(documents, n_questions, chain):
    questions = {}
    relevant_docs = {}
    for doc in tqdm.tqdm(documents, desc="Processing documents"):
        context = doc.page_content
        doc_id = doc.metadata["id"]

        # Generate questions for this document
        result = chain.invoke({"context": context, "n_questions": n_questions})
        generated_questions = result.content.strip().split('\n')

        # Process each generated question
        for q in generated_questions:
            # Remove the numbering from the question
            q = q.split('. ', 1)[-1].strip()

            # Generate a unique ID for this question
            question_id = str(uuid4())

            # Add the question to our questions dict
            questions[question_id] = q

            # Add the document ID to our relevant_docs dict
            relevant_docs[question_id] = [doc_id]

    return questions, relevant_docs