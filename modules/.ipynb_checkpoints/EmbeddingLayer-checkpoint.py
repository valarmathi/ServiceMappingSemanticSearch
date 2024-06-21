from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import openai

#Set API Key
filepath = 'config/OpenAPI_Key.txt'
with open(filepath, "r") as f:
  openai.api_key = ' '.join(f.readlines())

# Call PersistentClient()
client = chromadb.PersistentClient()

# Set up the embedding function using the OpenAI embedding model
model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model)

def generate_store_embedding_chromadb(document_pdfs_data):
    chroma_data_path = 'ChromaDB_Data'

    # Initialise a collection in chroma and pass the embedding_function to it so that it used OpenAI embeddings to embed the documents
    document_collection = create_collection('RAG_on_Documents')

    # Convert the page text and metadata from your dataframe to lists to be able to pass it to chroma
    documents_list = document_pdfs_data["Page_Text"].tolist()
    metadata_list = document_pdfs_data['Metadata'].tolist()

    # Add the documents and metadata to the collection alongwith generic integer IDs. You can also feed the metadata information as IDs by combining the policy name and page no.
    document_collection.add(
        documents = documents_list,
        ids = [str(i) for i in range(0, len(documents_list))],
        metadatas = metadata_list
    )
    
    return document_collection

def create_or_get_collection(collection_name):
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)