import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

load_dotenv()
huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_API_TOKEN")

current_dir = os.getcwd()

#Constructing file and directory paths relative to the current directory
file_path = os.path.join(current_dir, "books", "The_Constitution_of_Kenya_2010.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

#Displaying the paths
print("File path:", file_path)
print("Persistent directory path:", persistent_directory)


if not os.path.exists(persistent_directory):
  print("Persistent directory does not exist. Inializing vector store...")

  # Ensure the text file exists
  if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

  # Read the the text content from the file
  loader = PyPDFLoader(file_path)
  documents = loader.load()

  # Split the document into chunks
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs = text_splitter.split_documents(documents)

  # Display information about the split documents
  print("\n--- Document Chunks Information ---")
  print(f"Number of document chunks: {len(docs)}")
  print(f"Sample chunk: \n{docs[0].page_content}")

  # Create Embeddings
  print("\n --- Creating Embeddings ---")
  embeddings =  HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
  )
  print("\n ---Finished Creating Embeddings ---")

  # create the vector store and persist it automatically
  print("\n--- Creating vector store---")
  db = Chroma.from_documents(
      docs, embeddings, persist_directory=persistent_directory)
  print("\n--- Finished creating vector store ---")

else:
  print("Vector store already exists. No need to initialize.")


embeddings =  HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
  )

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# User Query
query = st.text_input("Ask the 2010 Constitution of Kenya something relevant")

#Retrieve relevant document sbased on the query
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":3, "score_threshold":0.1},
)

relevant_docs = retriever.invoke(query)

# Display the relevant information with metadata
#print("\n--- Relevant Documents ---")
#for i, doc in enumerate(relevant_docs, 1):
#  print(f"Document {i}:\n{doc.page_content}\n")
#  if doc.metadata:
#    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


model = HuggingFaceHub(
    repo_id="google/gemma-2b-it",
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512},
    huggingfacehub_api_token= "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"
)


# Define the prompt
prompt = f"""
<|system|>
You are an AI assistant that follows instructions extremely well.
Please be truthful and give direct answers.
Provide an answer based only on the provided documents.
If the answer is not found in the provided documents, report that you cannot answer based on the information provided.
</s>
<|user|>
Here are some documents that might help answer the question:
{query}

Relevant Documents:
{relevant_docs}
</s>
<|assistant|>
"""

# Generate response using the HuggingFaceHub model
response = model.predict(prompt)

# Extract the assistant's response
start_marker = "<|assistant|>\n"
end_marker = "</s>"

# Find the start and end positions of the assistant's response
start_pos = response.find(start_marker) + len(start_marker)
end_pos = response.find(end_marker, start_pos)

# Extract and print the response content
only_response = response[start_pos:end_pos].strip()
#print(only_response)
st.text(only_response)


#word = st.text_input("Phrase")
#st.write("The selected phrase is", word)
