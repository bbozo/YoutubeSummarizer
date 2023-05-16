from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', 'INSERT HERE')

loader = YoutubeLoader.from_youtube_url(sys.argv[1], add_video_info=True)
result = loader.load()

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(result)

chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts[:4])



