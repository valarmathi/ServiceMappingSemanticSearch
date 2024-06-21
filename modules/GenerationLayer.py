# Define the function to generate the response. Provide a comprehensive prompt that passes the user query and the top 3 results to the model
import openai

def generate_response(query, top_3_RAG):
    """
    Generate a response using GPT-3.5's ChatCompletion based on the user query and retrieved information.
    """
    messages = [
                {"role": "system", "content":  """You are a helpful assistant in analyzing ServiceNow documents who can effectively answer user queries about features provided by servicenow"""},
                {"role": "user", "content": f"""You are a helpful assistant in analyzing ServiceNow documents who can effectively answer user queries about features provided by servicenow by analyzing the documents given to you. You have a question asked by the user in '{query}' and you have some search results from a corpus of servicenow producr documents in the dataframe '{top_3_RAG}'. These search results are essentially one page of an product document that may be relevant to the user query.
The column 'documents' inside this dataframe contains the actual text from the product document and the column 'metadata' contains the feature name and source page. The text inside the document may also contain tables in the format of a list of lists where each of the nested lists indicates a row.
Use the documents in '{top_3_RAG}' to answer the query '{query}'. Frame an informative answer and also, use the dataframe to return the relevant document names and page numbers as citations.
Clearly follow the guidelines below when performing the task.
1. Try to provide relevant/accurate answers if available.
2. You don’t have to necessarily use all the information in the dataframe. Only choose information that is relevant.
3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular in format.
4. Use the Metadatas columns in the dataframe to retrieve and cite the document name(s) and Page No as citation. 
You should get the page no field and show relevant text eg:- Page 1 from the metadata. 
This page no. value is mandatory which you cannot skip and you can find the same from metadata.
5. If you can't provide the complete answer, please also provide any information that will help the user to search specific sections in the relevant cited documents.
6. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.

Follow the below chain of thoughts while answering the question.
1. User query: Are tag based services supported in domain separation?
2. Response from assistant: Yes, tag based services are supported in domain separation within ServiceNow. The documentation mentions about tag-based discovery capabilities within ServiceNow, which implies that tag based services are indeed supported in the context of ServiceNow's domain separation feature. 
**Citations:**', '1. Document Name: Tag-based discovery in ServiceNow', 'Page Number: Page 1'

The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. Follow this strictly to avoid giving irrelevant information to the user. If you couldn't find the answer, just return 'I couldn't find the information related to your query. Sorry for the inconvenience'. Provide the final response as a well-formatted and easily readable text along with the citation. Provide your complete response first with all information, and then provide the citations."""}
              ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.split('\n')
