from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

LOCATION = ":memory:"
COLLECTION_NAME = "Ethical AI RAG"
VECTOR_SIZE = 384

PDF_URLS = [
    'https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf',
    'https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf'
]

UNTRAINED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

base_rag_prompt_template = """\
Only answer questions that are related to the context provided. DO NOT answer a question if it is unrelated to the context provided. If you do not know the answer, say "I do not know the answer".

Context:
{context}

Question:
{question}
"""

BASE_RAG_PROMPT = ChatPromptTemplate.from_template(base_rag_prompt_template)

BASE_LLM = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"])

METRICS = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
]
