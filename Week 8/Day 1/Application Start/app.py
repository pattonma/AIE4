### Import Section ###
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
import uuid
import chainlit as cl

### Global Section ###
chat_model = ChatOpenAI(model="gpt-4o-mini")
set_llm_cache(InMemoryCache())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""
rag_message_list = [{"role" : "system", "content" : rag_system_prompt_template},]
rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""
chat_prompt = ChatPromptTemplate.from_messages([("system", rag_system_prompt_template), ("human", rag_user_prompt_template)])
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(collection_name=collection_name,vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings, store, namespace=core_embeddings.model)
vectorstore = QdrantVectorStore(client=client,collection_name=collection_name,embedding=cached_embedder)
Loader = PyMuPDFLoader

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin.",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="No file was uploaded. Please try again.").send()
        return

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Save the file locally
    with open(file.name, "wb") as f:
        f.write(file.content)

    # Load and process the document
    loader = Loader(file.name)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    # Add documents to the vectorstore
    vectorstore.add_documents(docs)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create RAG chain
    global retrieval_augmented_qa_chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt
        | chat_model
    )

    await cl.Message(content=f"`{file.name}` processed. You can now ask questions about its content.").send()

    

### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    return "AI Assistant"

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    response = retrieval_augmented_qa_chain.invoke({"question": message.content})
    await cl.Message(content=response.content).send()