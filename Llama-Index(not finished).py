# We are going to use Retrieval Augmented Generation (RAG) instead of fine tuning to response to the user's query directly.
#%%
%pip install utils
%pip install openai
%pip install llama-index
%pip install pypdf
#%%
%pip install llama_index
#%%
import utils
import os
import openai
openai_api_key="********"
# %%
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["FineTuning-Datasheet-Demo\ddd.pdf"]
).load_data()
# %%
print(documents)
# %%
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print("############")
print(documents[0].text)
# %%
#Using basic RAG pipeline

from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))
print(documents)
# %%
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
#%%
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model = "local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
service_context=service_context)
# %%
#Query Engine

query_engine = index.as_query_engine()

response = query_engine.query(
    "What is the maximum current allowed in this device?"
)

# Print the text content of the response
print(response.text)
# %%
