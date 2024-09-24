(Please note that this is just the written portion. All of the information in this file is also available alongside relevant code blocks in the [midterm_fullcode.ipynb]())

## Task 1: Dealing with the Data

The default chunking strategy I will use for the RAG prototype will be recursive text splitting. This is a method that is actually recommended by LangChain themselves as a quick and simply way to start splitting documents, which is exactly what I'm after for this first quick RAG prototype. Mechanically, this strategy uses a list of separators and applies a series of splits in order through that list, resulting is a coarse-to-fine-grained splitting approach. If a chunk that it returns from a coarser split is too large, it recursively splits it using a "finer-grained" separator. This strategy results in more logically coherent and semantically meaningful chunks than a simpler strategy like fixed-size chunking, which is useful during our retrieval process later. This approach is also fairly tunable, allowing me to pick a chunk size and overlap very easily.

Associated code:
```python
recursiveChunker = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 60,
    length_function = len,
)
recursive_split_docs = recursiveChunker.split_documents(all_documents)
```


My second approach to chunking will be semantic chunking. This is a more advanced chunking strategy that splits chunks based on their semantic similarity. If embeddings of the document text are found to be sufficiently far apart, they are split into separate chunks. The goal of semantic chunking is to preserve as much coherence as possible in the individual chunks. This is also helpful because it is less reliant on the documents themselves being simply large blocks of text than the recursive  splitter. In this particular case, this is very helpful for the NIST paper, which has large sections of text in a tabular format, which may affect the recursive splitter. The downside of semantic chunking is that it is much more computationally intensive than recursive text-splitting, requiring an embedding model to be used during the chunking process itself. However, it has the possibility to be a much better chunking strategy for a production-level application.

Associated code:
```python
semanticChunker = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile"
)
semantic_split_docs = semanticChunker.split_documents(all_documents)
```


Why I have chosen these two specific chunking strategies essentially comes down to ease of implementation for this prototype. The recursive text splitting is very quick and simple to implement. It should answer any of the simple and moderately difficult questions that the stakeholders may ask it. For the simple proof-of-concept that will be shown to the SVP, this should suffice. The second strategy of semantic chunking was chosen because of the potential increase in chunk quality over the simpler recursive strategy. It should result in a much better set of chunks for the retrievers to work with. (Also, as Mark and not the AI Solutions Engineer, I just want to try it!) This should allow the production-level RAG application to answer much more detailed and difficult questions from stakeholders.

## Task 2: Building a Quick End-to-End Prototype

Full code and prototype application hosted seperately at [this HuggingFace repo](https://huggingface.co/spaces/pattonma/AIE4_Midterm_Prototype_RAG).

Loom video showing demo of the prototype found [here](https://www.loom.com/share/4776ab3cd810434ba787c7b1b05998a1).

The prototype, at a high level, is a Langchain RAG application utilizing chainlit, which is then dockerized and hosted on a Hugging Face space for ease of access. Here's a detailed breakdown of the application stack, and why each piece was chosen:

1. **Application Framework:**

    I chose Chainlit to manage the chatbot interface and real-time messaging. The reason I picked Chainlit over something like Streamlit is that Chainlit is purpose-built for building chat-based LLM apps. I'm also more familiar with it as a tool than I am alternatives.

2. **Document Processing:**

    Document Loader (`PyMuPDFLoader`): I use `PyMuPDFLoader` from LangChain’s community module to load and parse PDFs. I selected it because it efficiently handles PDF loading and text extraction, and it's well-integrated with LangChain. While alternatives like `pdfplumber` or `PyPDF2` exist, once again I am more familiar with PyMuPDFLoader.

    Text Splitting (`RecursiveCharacterTextSplitter`): After loading the documents, I split the text into chunks of 600 characters using the RecursiveCharacterTextSplitter. This choice was made over something like a simple `CharacterTextSplitter` because `RecursiveCharacterTextSplitter` is much better at handling sentences, as chunks are broken at meaningful points (e.g., sentence boundaries) rather that strcit character counts, which improves the relevance of the retrieved context for the RAG pipeline.

3. **Vector Store and Embedding Model:**

    Qdrant Vector Store (`QdrantVectorStore`): I decided to use Qdrant as my vector store. While I am personally more familiar with Pinecone, I chose Qdrant for its seamless integration with LangChain and its built-in support for features like the ability to store vectors in memory. Since this app is hosted on Hugging Face Spaces, I opted for an in-memory store `(:memory:)` to minimize resource use.

    Embeddings Model (`all-MiniLM-L6-v2`): For generating document embeddings, I used the sentence-transformers/all-MiniLM-L6-v2 model from Hugging Face. I chose this model over something like OpenAI's embedding models due to its lower cost (it's open-source), its efficiency, and the fact that it performs well for a wide range of tasks. It strikes a balance between speed and accuracy, which makes it suitable for real-time applications like this one. And because it's open-source, I have the ability to fine-tune it for our specific application.

4. **Large Language Model (LLM):**

    LLM (ChatOpenAI): I use the `GPT-4o-mini` model as my LLM for generating responses. I opted for 4o-mini because it is a competent and cheap LLM. This application does not need a sophisticated LLM to analyze retrieved context and answer the questions that stakeholders may ask it. Also, OpenAI models are well integrated into the Langchain ecosystem.

5. **Retrieval-Augmented Generation (RAG) Pipeline:**

    Langchain RAG Chain: The `RetrievalAugmentedQAPipeline` is constructed using LangChain’s built-in constructs for combining document retrieval and answer generation. Langchain allows easy customizition and combination of different modules, tailored for our specific application. The pipeline first retrieves relevant document chunks from the Qdrant vector store based on the user's question. I opted for LangChain's built-in chain components because they allow for a flexible, modular design while abstracting away many complexities, such as managing how retrieved documents are passed along the chain.

6. **Hugging Face Integration:**

    Hosted on Hugging Face Spaces: I chose Hugging Face Spaces for deployment because it's an ideal platform for hosting NLP applications with minimal setup. Spaces provide a pre-built environment for running applications with GPU support, which can speed up LLM inference. Hosting on Hugging Face also allows the stakeholders to access the applicaiton from the publicly available web, rather than us hosting the application on our private local network, as who knows where in the world the stakeholders are.

7. **Dockerization (Hosting on Hugging Face Spaces):**

    I dockerized the application to ensure a consistent runtime environment. I chose Docker because it provides containerization, which guarantees that the application runs the same way in any environment. This also makes it easy to handle dependencies, especially when deploying the app to Hugging Face Spaces, which support Docker out of the box.


## Task 3: Creating a Golden Test Data Set

After evaluating my default RAG chain, using the default chunker (`RecursiveCharacterTextSplitter`) and the untrained `all-MiniLM-L6-v2` model, I have found it results in scores for the following metrics:
| Metric | Score |
| :------- | ------:|
| Faithfulness | 0.8371 |
| Answer Relevancy | 0.8621 |
| Context Recall | 0.7498 |
| Context Precision | 0.8878 |

Based on those scores, I can draw the following conclusions:
1. **Strong Relevance and Precision**: The pipeline is generally effective in generating relevant answers (with an answer relevancy score of 0.8621) and retrieving relevant document chunks (context precision of 0.8878). This suggests that our model’s retrieval and generation components are functioning well, but not perfectly optimized.

2. **Recall Gap**: The relatively lower context recall (0.7498) compared to the other metrics indicates that the retrieval process is missing some relevant information. This suggests that our default chunking strategy and untrained embedding model are not fully optimized, and some important context might be lost or not retrieved.

3. **Room for Improvement in Faithfulness and Recall**:
    - The faithfulness score of 0.8371 indicates that while answers are generally grounded in the retrieved context, there may still be hallucinations or deviations from the content, as ~16% of the generated answers may contain information that isn’t strictly derived from the retrieved documents.
    - A context recall of 0.7498 suggests that improving the retrieval step (through better embeddings or chunking) could enhance the pipeline’s ability to pull more relevant information, which could also boost faithfulness and answer relevancy.

## Task 4: Fine-Tuning Open-Source Embeddings

A link to my trained embeddings can be found [on Hugging Face](https://huggingface.co/pattonma/AIE4_midterm_tuned_embeddings).

I chose to use the model `sentence-transformers/all-MiniLM-L6-v2` because it is an open source embedding model and it is widely downloaded off of Hugging Face, indicating to me that it is a generally well-performing and easy-to-train model. It's also intended for shorter paragraphs and sentences, which coincides well with the documents we're concerned with at the moment. Neither of the docuements have particularly long, uninterrupted strings of text. Being tuned for sentences also makes it very good with semantic searches, which are highly relevant for Q+A tasks. 

It's also a fairly small and compact model, with only ~23m paramters. This helps it strike a good balance between accuracy and speed, making it well-suited for our application, which only makes use of a few documents. Also, being small, it's quick and easy to tune on even modest hardware (no A100 required!), which is perfect for getting our prototype up and running quickly. Similar to the parameters, the actual embedding vector sizes are also fairly small, being only 384 dimensions. This makes storing them in our vector store (Qdrant) take less storage (good because we're just running Qdrant's storage in memory) and the searches quick (because fewer vectors means quicker similarity searches). Overall, it's not powerful, but it gets the job done.

## Task 5: Assessing Performance

Testing all combinations of the Chunking Strategies (Recursive, Semantic) and Embedding Models (Untrained, Trained):

| Metric| Recursive+Untrained | Recursive+Trained | Semantic+Untrained | Semantic+Trained |
|:--|:-:|:-:|:-:|:-:|
| faithfulness | 0.837098 | 0.830230 | 0.900540 | 0.894261 |
| answer_relevancy | 0.862088 | 0.868417 | 0.848056 | 0.909716 |
| context_recall | 0.749788 | 0.739153 | 0.887333 | 0.896095 |
| context_precision | 0.887778 | 0.899444 | 0.901111 | 0.905556 |

Given the results of the various evaluations we've done, I would recommend using the SemanticChunker and Trained Embeddings combination for our production chain. 

It is all around the best option, as it is the best combination in terms of nearly every metric, except faithufulness, which is essentially tied with the Semantic+Untrained chain. Its high faithfulness indicates this combination generates highly accurate responses based on the retrieved content. Its high answer relevancy means that the responses generated are extremely relevant to the user's questions. This metric is critical for a Q+A system because it directly reflects how well the model understands and responds to the user's query. The high context recall shows this combination is retrieving more relevant chunks that other chains, and the high context precision also shows that the chunks it does retrieve are highly relevant to the query.

Why Not the Other Combinations:
1. Recursive Chunking + Untrained/Trained Embeddings:
    These approaches have lower scores in faithfulness and context recall, likely because the simpler chunking method is splitting important information or mixing unrelated content. This reduces the system’s ability to retrieve complete, meaningful chunks.
2. Semantic Chunking + Untrained Embeddings:
    While this combination retrieves faithful information, the untrained embeddings reduce its ability to fully understand the queries (lower answer relevancy).

## Task 6: Managing Your Boss and User Expectations

The story I would give to the CEO to communicate with the rest of the company would be a company-wide email, or press release, along the lines of:

> ### Introducing Our Ethical AI Chatbot: Guiding the Future of AI in Enterprise
>
>Over the past several months, our technology team has been working tirelessly to address one of the most pressing and complex challenges we face as a company: **How do we build AI solutions that are not only powerful but also ethical and aligned with our company’s values?**
>
>Through our conversations with various internal stakeholders, it became clear that there is **a growing concern around the implications of AI**—especially as we navigate an election cycle that will inevitably influence AI regulation and policy. Many of you expressed **a need to better understand the evolving landscape of AI**, particularly as it relates to government policy and regulations that are likely to shape the future of our industry. And let’s be honest—keeping up with the pace of change in AI is hard. There are new developments every day, and many of them have the potential to reshape the way we operate, the products we build, and the services we provide.
>
>With this feedback in mind, we took action. Today, I’m proud to introduce a new **AI-powered chatbot** that will help all of us better understand the evolving AI landscape, particularly how it intersects with politics, regulation, and ethical considerations. This chatbot has been designed to provide clarity around these topics using critical and pertient documents, straight from the experts who may end up writing the laws that dictate AI.
>
>These documents are the cornerstone of our chatbot’s knowledge base. Our team has developed a Retrieval-Augmented Generation (RAG) system, which uses these documents as the primary source of truth for answering your questions about AI ethics, government policies, and industry best practices.
>
>What sets this tool apart is that it’s not just a simple chatbot. We’ve fine-tuned it to our sepcific use case to ensure that the answers you receive are both relevant and grounded in the most critical, authoritative texts on the subject. Whether you’re asking about how AI regulations could impact our business or what ethical considerations we should be thinking about as we build new AI-powered tools, this chatbot is designed to provide thoughtful, reliable responses.
>
> #### Why This Matters Now
>This is more than just a tech solution—it’s part of our commitment to being a leader in the ethical deployment of AI. As we move forward, there will be political, social, and ethical questions that we will need to address head-on. This chatbot will help us as a company to navigate the evolving AI landscape, giving each of you the knowledge and tools to better understand how AI policies are being shaped—and how we can shape our own AI initiatives responsibly.
>
> #### How You Can Help Shape the Future
>In the next month, our AI Solutions team will be working with 50 internal stakeholders to test and refine this tool based on your feedback. This is an evolving project, and your input will be invaluable in ensuring the chatbot is not just answering questions, but truly helping guide our approach to ethical AI development. After this test phase, we’ll be making the chatbot available across the entire company.
>
> #### Moving Forward
>This is just the beginning. Our goal is to not only provide a tool that educates and informs but to spark conversations within the company about how we can ensure our AI systems are designed, deployed, and managed in ways that are aligned with our company’s values—and ultimately, the values of the society we serve.
>
>I encourage you all to engage with the chatbot once it’s rolled out, to ask tough questions, and to use it as a resource in your daily work. Together, we will ensure that our company remains at the forefront of responsible AI development, and that we are equipped to navigate the rapidly changing AI landscape with confidence and clarity.
>
>Thank you for your continued support, and I look forward to seeing how this tool will empower us all to lead in the future of AI.

Considerations for future updates, such as an updated or expanded list of pertinent documents (more White House briefs, executive orders, Nonprofit research papers, etc), we'd just have to modify the list of documents that we load at the start of the application. Ideally, we'd actually store them in a dedicated vector store, rather than reading them and storing them in memory at runtime, particularly if the list of context documents that we're working with ends up being quite large. 

Other considerations for the future may include changing to a different embedding model (perhaps one with more parameters or a larger embedding dimension). This would be beneficial if, again, our list of documents started getting large. Further explorations into new chunking strategies, better fine-tuning (many hyper paramters can be changed that were not when I tuned my embeddings), better prompting of the chains (all prompts at the moment are very basic), and even UI improvements would be warranted. 

One change that I honestly don't know if it's warranted or not would be to implement a fine tuned LLM for the chains, as the simple Q+A nature of this application doesn't need a particularly powerful or specialized model (which is not to say that a model tuned to Q+A wouldn't be beneficial, but it seems like it may be a diminishing return depending on the time invested versus performance improvements it may yield). But, depending on how widely used this application may become, it might behoove the company to change from using the proprietary LLMs that it uses now and move to open-source models, just as a smiple cost-cutting measure. That may also provide at least an opening to the opportunity of fine-tuning the model for Q+A purposes.