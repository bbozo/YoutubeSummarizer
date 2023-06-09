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

loader = YoutubeLoader.from_youtube_url(sys.argv[1], add_video_info=True)
result = loader.load()

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(result)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

# -6901589.svc.asia-southeast1-gcp-free.pinecone.io
index_name = "langchaintest" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
