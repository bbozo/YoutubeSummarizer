from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import sys
import os


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'INSERT HERE')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'INSERT HERE')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'INSERT HERE')

arguments_as_string = ' '.join(sys.argv[1:])


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

index_name = "langchaintest" # put in the name of your pinecone index here

docsearch = Pinecone.from_existing_index(index_name, embeddings)

chain = load_qa_chain(llm, chain_type="stuff")

query = arguments_as_string
docs = docsearch.similarity_search(query)

print(chain.run(input_documents=docs, question=query))


