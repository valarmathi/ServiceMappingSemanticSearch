from modules.PDFExtractionUtil import extract_data_from_directory
from modules.EmbeddingLayer import generate_store_embedding_chromadb
from modules.EmbeddingLayer import create_or_get_collection
from modules.SearchLayer import do_semantic_search
from modules.GenerationLayer import generate_response
from pathlib import Path
import pandas as pd
import openai

# Define the path where all pdf documents are present
pdf_path = "data"

class SemanticSearchBot:
    document_pdfs_data = None
    
    def generate_data_frame(self):
        # Define the directory containing the PDF files
        pdf_directory = Path(pdf_path)
        
        data = extract_data_from_directory(pdf_directory)

        # Concatenate all the DFs in the list 'data' together
        document_pdfs_data = pd.concat(data, ignore_index=True)

        #filter based on document length so that we can ignore empty pages
        document_pdfs_data['Text Length'] = document_pdfs_data['Page_Text'].apply(lambda x: len(x.split(' ')))
        document_pdfs_data = document_pdfs_data[document_pdfs_data['Text Length'] > 20]

        #Add metadata to the dataframe
        document_pdfs_data['Metadata'] = document_pdfs_data.apply(lambda x: {'Document Name':x['Document Name'],'Page No':x['Page No.']}, axis=1)
        self.document_pdfs_data = document_pdfs_data

    def store_embeddings_in_chroma(self):
        self.generate_data_frame()

        #Layer 1
        document_collection = generate_store_embedding_chromadb(self.document_pdfs_data)

    def initialize_chatbot(self):
        #Layer 2 - Semantic search with cache
        
        # Read the user query
        print('Please enter an user query to proceed')
        query = input()

        # Searh the Cache collection first
        # Query the collection against the user query and return the top 5 results
        top_3_RAG = do_semantic_search(query)
        #print(top_3_RAG.to_string())

        #Layer 3 - Retrieval Augmented Generation
        response = generate_response(query, top_3_RAG)
        print(response)


        
        