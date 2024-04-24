from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
import tablehelper as tb
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex 
from azure.search.documents.indexes.aio import SearchIndexClient
import asyncio
import requests
import openai
import os
from openai import AzureOpenAI 

openai.api_type = "azure"
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_base = "https://opai-genaisppi-dev-swe-001.openai.azure.com"
openai.api_version = "2023-05-15"

AzureOpenAIclient =AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_KEY"),    api_version = "2023-05-15",
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)
response = AzureOpenAIclient.embeddings.create(
    input="I like to eat apples and bananas.",
    model="text-embedding-ada-002")
embeddings = response.data[0].embedding
print(embeddings)
print(response.model_dump_json(indent=2))


# os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
# os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
# os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
# os.environ["OPENAI_API_TYPE"] = "azure"
# #embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002",model="text-embedding-ada-002", openai_api_base= os.environ["OPENAI_API_BASE"],openai_api_type="azure",openai_api_key = os.environ["OPENAI_API_KEY"],chunk_size=16)

# embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002', openai_api_version="2023-05-15")
# embedding = embeddings.embed_documents(["I like to eat apples and bananas."])
