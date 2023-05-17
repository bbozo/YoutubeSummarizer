from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os


def build_prompt(header, summary="ELABORATE SUMMARY"):
    template = f"""{header} of the following:
    
    
    {{text}}
    
    {summary}:
    """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    return prompt


OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', 'INSERT HERE')

loader = YoutubeLoader.from_youtube_url(sys.argv[1], add_video_info=True)
result = loader.load()

llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(result)

chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, map_prompt=build_prompt("give a concise summary", summary="CONCISE SUMMARY"), combine_prompt=build_prompt("give an elaborate 10 point summary"))
print(chain.run(texts[:4]))



