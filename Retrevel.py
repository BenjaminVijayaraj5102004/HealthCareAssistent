from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the Excel file
df = pd.read_excel("medicine_suggestions.xlsx")


# Initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# Define the database location
db_location = "./chroma_lanchain_db"

# Check if the database exists
add_document = not os.path.exists(db_location)

if add_document:
    documents = []
    ids = []
    
    # Iterate over rows in the DataFrame
    for i, row in df.iterrows():
        # Convert all values to strings before concatenation
        document = Document(
            page_content=(
                str(row["Symptom(s)"]) + " " + 
                str(row["Possible Condition"]) + " " + 
                str(row["Medicine Name"]) + " " + 
                str(row["Dosage"]) + " " + 
                str(row["When to Use"]) + " " + 
                str(row["Duration (Days)"]) + " " + 
                str(row["Notes"])
            ),
            metadata={"row_id": i}  # Store row ID as metadata
        )
        documents.append(document)
        ids.append(str(i))

# Initialize the vector store
vector_store = Chroma(
    collection_name="medicine_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents if necessary
if add_document:
    vector_store.add_documents(documents=documents, ids=ids)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)
