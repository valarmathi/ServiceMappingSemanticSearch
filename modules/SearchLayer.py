from modules.EmbeddingLayer import create_or_get_collection
import openai
import pandas as pd
from sentence_transformers import CrossEncoder, util

# Implementing Cache in Semantic Search
def do_semantic_search(query):
    n_results=5
    document_collection = create_or_get_collection('RAG_on_Documents')
    
    cache_collection = create_or_get_collection('Document_Cache')
    cache_results = cache_collection.query(
        query_texts=query,
        n_results=n_results
    )
    
    # Set a threshold for cache search
    threshold = 0.2
    
    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()
    
    # If the distance is greater than the threshold, then return the results from the main collection.
    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
          # Query the collection against the user query and return the top 10 results
          results = document_collection.query(
              query_texts=query,
              n_results=5
          )

          # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
          # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
          Keys = []
          Values = []
    
          for key, val in results.items():
            if val is None or key=='included':
              continue
            for i in range(n_results):
              Keys.append(str(key)+str(i))
              Values.append(str(val[0][i]))
    
          cache_collection.add(
              documents= [query],
              ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
              metadatas = dict(zip(Keys, Values))
          )
    
          print("Not found in cache. Found in main collection.")
    
          result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
          results_df = pd.DataFrame.from_dict(result_dict)
          results_df
    
    # If the distance is, however, less than the threshold, you can return the results from cache
    elif cache_results['distances'][0][0] <= threshold:
          cache_result_dict = cache_results['metadatas'][0][0]
    
          # Loop through each inner list and then through the dictionary
          for key, value in cache_result_dict.items():
              if 'ids' in key:
                  ids.append(value)
              elif 'documents' in key:
                  documents.append(value)
              elif 'distances' in key:
                  distances.append(value)
              elif 'metadatas' in key:
                  metadatas.append(value)
    
          print("Found in cache!")
    
          # Create a DataFrame
          results_df = pd.DataFrame({
            'IDs': ids,
            'Documents': documents,
            'Distances': distances,
            'Metadatas': metadatas
          })

    return rerank_documents_with_cross_encoder(query, results_df)

def rerank_documents_with_cross_encoder(query, results_df):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Input (query, response) pairs for each of the top 5 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs
    #cross encoder accepts nested array of 2 sentences to find the similarity between the 2
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)

    results_df['Reranked_scores'] = cross_rerank_scores

    # Return the top 3 results after reranking
    top_3_rerank = results_df.sort_values(by='Reranked_scores', ascending=False)
    top_3_rerank = top_3_rerank[:3]

    top_3_RAG = top_3_rerank[["Documents", "Metadatas"]]
    
    return top_3_RAG
        