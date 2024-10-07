### Import Section ###
from google.colab import files
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
    """ SESSION SPECIFIC CODE HERE """
    # get file
    #process file
    # creatre vectorstore
    #create retriever
    #create chain
    

### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """
    MESSAGE CODE HERE
    """