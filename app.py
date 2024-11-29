# Import required libraries
import gradio as gr
from pymilvus import MilvusClient
import numpy as np
import requests
from mixedbread_ai.client import MixedbreadAI
from dotenv import dotenv_values
import re
from functools import cache
import pandas as pd

################################################################################
# Configuration

# Define Milvus client
milvus_client = MilvusClient("http://localhost:19530")

# Import secrets
config = dotenv_values(".env")

# Setup mxbai
mxbai_api_key = config["MXBAI_API_KEY"]
mxbai = MixedbreadAI(api_key=mxbai_api_key)

################################################################################
# Function to medrXiv DOI from input text
def extract_doi(text):

    # Define regex pattern
    pattern = re.compile(r"10\.1101\/(?:\d{4}\.\d{2}\.\d{2}\.)?\d{8}")

    # Search for matches
    match = pattern.search(text)

    # Return the match if found, otherwise return None
    return match.group(0) if match else None

# Function to search MedRiv by DOI
@cache
def fetch_medrxiv_by_id(doi):

    # Define the base URL for the medRxiv API
    base_url = "https://api.medrxiv.org/details/medrxiv/"

    # Construct the full URL for the API request
    url = f"{base_url}{doi}"

    # Send the API request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200 and response.json()['messages'][0]['status'] == 'ok':

        # Parse the JSON response
        data = response.json()

        # Extract the abstract from the response data
        return {
            "Title": data['collection'][0]['title'],
            "Authors": data['collection'][0]['authors'],
            "Abstract": data['collection'][0]['abstract'],
            "URL": f"https://doi.org/{doi}"
        }
    else:
        # Raise an exception if the request was not successful
        raise gr.Error( f"Failed to fetch metadata for DOI {doi}. {response.json()['messages'][0]['status']}")

################################################################################
# Function to embed text
@cache
def embed(text):

    res = mxbai.embeddings(
    model='mixedbread-ai/mxbai-embed-large-v1',
    input=text,
    normalized=True,
    encoding_format='float',
    truncation_strategy='end'
    )

    vector = np.array(res.data[0].embedding)

    return vector

################################################################################
# Single vector search

def search(vector, limit):

    result = milvus_client.search(
        collection_name="medrxiv_abstracts", # Replace with the actual name of your collection
        # Replace with your query vector
        data=[vector],
        limit=limit, # Max. number of search results to return
        search_params={"metric_type": "COSINE"}, # Search parameters
        output_fields=["$meta"] # Output fields to return
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]

################################################################################
# Function to fetch paper details of all results
def fetch_all_details(search_results):

    all_details = []

    for search_result in search_results:

        paper_details = search_result['entity']

        paper_details['Similarity Score'] = np.round(search_result['distance']*100, 2)

        all_details.append(paper_details)

    # Convert to dataframe
    df = pd.DataFrame(all_details)

    # Make a card for each row
    cards = ""

    for index, row in df.iterrows():

    # chr(10) is a new line character, replace to avoid formatting issues
        card = f"""
## [{row["Title"].replace(chr(10),"")}]({row["URL"]})
> {row["Authors"]} \n
{row["Abstract"]}
***
"""
    
        cards +=card
    
    return cards

################################################################################

# Function to handle the UI logic
@cache
def predict(input_text, limit=5, increment=5):

    # Check if input is empty
    if input_text == "":
        raise gr.Error("Please provide either an MedRxiv DOI or an abstract.", 10)  

    # Define extra outputs to pass
    # This hack shows the load_more button once the search has been made
    show_element = gr.update(visible=True)

    # This variable is used to increment the search limit when the load_more button is clicked
    new_limit = limit+increment
    
    # Extract MedRxiv DOI, if any
    doi = extract_doi(input_text)

    # When medRxiv doi is found in input text
    if doi:

        # Search if id is already in database
        id_in_db = milvus_client.get(collection_name="medrxiv_abstracts",ids=[doi])

        # If the id is already in database
        if bool(id_in_db):

            # Get the vector
            abstract_vector = id_in_db[0]['vector']

        # If the id is not already in database
        else:

            # Search medRxiv for paper details
            medrxiv_json = fetch_medrxiv_by_id(doi)

            # Embed abstract
            abstract_vector = embed(medrxiv_json['Abstract'])
    
    # When medRxiv doi is not found in input text, treat input text as abstract
    else:

        # Embed abstract
        abstract_vector = embed(input_text)

    # Search database
    search_results = search(abstract_vector, limit)

    # Gather details about the found papers
    all_details = fetch_all_details(search_results)
        
    return all_details, show_element, new_limit
            
################################################################################

# Variable to store contact information
contact_text = """
<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
    <h3>Crafted with ❤️ by <a href="https://www.linkedin.com/in/mitanshusukhwani/" target="_blank">Mitanshu Sukhwani</a></h3>
    <h4>Discover more at <a href="https://papermatch.mitanshu.tech" target="_blank">PaperMatch</a></h4>
</div>
"""

# Examples to display
examples = [
    "10.1101/2019.12.08.19013979",
    "Game theory applications in Medicine"
]


# Show total number of entries in database
num_entries = format(milvus_client.get_collection_stats(collection_name="medrxiv_abstracts")['row_count'], ",")

# Create a back to top button
back_to_top_btn_html = '''
<button id="toTopBtn" onclick="'parentIFrame' in window ? window.parentIFrame.scrollTo({top: 0, behavior:'smooth'}) : window.scrollTo({ top: 0 })">
    <a style="color:#6366f1; text-decoration:none;">&#8593;</a> <!-- Use the ^ character -->
</button>'''

# CSS for the back to top button
style = """
#toTopBtn {
    position: fixed;
    bottom: 10px;
    right: 10px; /* Adjust this value to position it at the bottom-right corner */
    height: 40px; /* Increase the height for a better look */
    width: 40px; /* Set a fixed width for the button */
    font-size: 20px; /* Set font size for the ^ icon */
    border-color: #e0e7ff; /* Change border color using hex */
    background-color: #e0e7ff; /* Change background color using hex */
    text-align: center; /* Align the text in the center */
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 50%; /* Make it circular */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Add shadow for better visibility */
}

#toTopBtn:hover {
    background-color: #c7d4ff; /* Change background color on hover */
}
"""

################################################################################
# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(font=gr.themes.GoogleFont("Helvetica"), 
                                    font_mono=gr.themes.GoogleFont("Roboto Mono")), 
                                    title='PaperMatchMed', css=style) as demo:

    # Title and description
    gr.HTML('<h1><a href="https://papermatchmed.mitanshu.tech" style="font-weight: bold; text-decoration: none;">PaperMatchMed</a></h1>')
    gr.Markdown("### Discover Relevant Research, Instantly ⚡")
    
    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            placeholder=f"Search {num_entries} papers on medRxiv",
            autofocus=True,
            submit_btn=True,
            show_label=False
        )
    
    # Define the initial page limit
    page_limit = gr.State(5)

    # Define the increment for the "Load More" button
    increment = gr.State(5)

    # Define new page limit
    new_page_limit = gr.State(page_limit.value + increment.value)

    # Output section, displays the search results
    output = gr.Markdown(label="Related Papers", latex_delimiters=[{ "left": "$", "right": "$", "display": False}])

    # Hidden by default, appears after the first search
    load_more_button = gr.Button("More results ⬇️", visible=False)

    # Event handler for the input text box, triggers the search function
    input_text.submit(predict, [input_text, page_limit, increment], [output, load_more_button, new_page_limit])

    # Event handler for the "Load More" button
    load_more_button.click(predict, [input_text, new_page_limit, increment], [output, load_more_button, new_page_limit])

    # Example inputs
    gr.Examples(
        examples=examples, 
        inputs=input_text,
        outputs=[output, load_more_button, new_page_limit],
        fn=predict,
        label="Try:",
        run_on_click=True)

    # Back to top button
    gr.HTML(back_to_top_btn_html)

    # Attribution
    gr.HTML(contact_text)


################################################################################

if __name__ == "__main__":
    demo.launch(server_port=7862, favicon_path='logo.png')
