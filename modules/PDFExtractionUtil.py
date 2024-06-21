import pdfplumber
from operator import itemgetter
import pandas as pd
import json
import tiktoken

# Function to check whether a word is present in a table or not for segregation of regular text and tables
def check_bboxes(word, table_bbox):
    # Check whether word is inside a table bbox.
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]



# Function to extract text from a PDF file.
# 1. Declare a variable p to store the iteration of the loop that will help us store page numbers alongside the text
# 2. Declare an empty list 'full_text' to store all the text files
# 3. Use pdfplumber to open the pdf pages one by one
# 4. Find the tables and their locations in the page
# 5. Extract the text from the tables in the variable 'tables'
# 6. Extract the regular words by calling the function check_bboxes() and checking whether words are present in the table or not
# 7. Use the cluster_objects utility to cluster non-table and table words together so that they retain the same chronology as in the original PDF
# 8. Declare an empty list 'lines' to store the page text
# 9. If a text element in present in the cluster, append it to 'lines', else if a table element is present, append the table
# 10. Append the page number and all lines to full_text, and increment 'p'
# 11. When the function has iterated over all pages, return the 'full_text' list

def extract_text_from_pdf(pdf_path):
    p = 0
    full_text = []


    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = f"Page {p+1}"
            text = page.extract_text()

            tables = page.find_tables()
            table_bboxes = [i.bbox for i in tables]
            tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
            non_table_words = [word for word in page.extract_words() if not any(
                [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
            lines = []

            for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5):

                if 'text' in cluster[0]:
                    try:
                        lines.append(' '.join([i['text'] for i in cluster]))
                    except KeyError:
                        pass

                elif 'table' in cluster[0]:
                    lines.append(json.dumps(cluster[0]['table']))


            full_text.append([page_no, " ".join(lines)])
            p +=1

    return full_text

def extract_data_from_directory(pdf_directory):
    # Initialize an empty list to store the extracted texts and document names
    data = []
    
    # Loop through all files in the directory
    for pdf_path in pdf_directory.glob("*.pdf"):
    
        # Process the PDF file
        print(f"...Processing {pdf_path.name}")
    
        # Call the function to extract the text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)
    
        # Convert the extracted list to a PDF, and add a column to store document names
        extracted_text_df = pd.DataFrame(extracted_text, columns=['Page No.', 'Page_Text'])
        extracted_text_df['Document Name'] = pdf_path.name
    
        # Append the extracted text and document name to the list
        data.append(extracted_text_df)
    
        # Print a message to indicate progress
        print(f"Finished processing {pdf_path.name}")
    
    # Print a message to indicate all PDFs have been processed
    print("All PDFs have been processed.")

    return data