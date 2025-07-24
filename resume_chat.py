import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# setup gemini api key
api_key = os.environ["GOOGLE_API_KEY"]

# setup llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# load resume and split
loader = PyPDFLoader("RESUME_RVirtus.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create embeddings and index intro FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# fine tune prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a helpful resume assistant.
        You answer questions based **only** on the following resume information:

        {context}

        Question: {question}
        Answer:"""
    )

# setup retriever chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# chat
print("\n Chat with your resume!")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    response = qa_chain.run(question)
    print("AI:", response, "\n")