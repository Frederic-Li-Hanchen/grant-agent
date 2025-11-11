import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
from time import sleep, time
import random  # For random sleep intervals
from typing import Optional
from utils import load_config_from_yaml
from pdb import set_trace as st
from dotenv import load_dotenv
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel, Field
import vertexai
from typing import List, Dict, Any
import json
import sys
import pandas as pd



# =================================================
# === Building the database of Bekanntmachungen ===
# =================================================

### Helper function that checks if a webpage exists
# Input:
# - [str] url: the URL of the webpage to check
# Output:  
# - [bool] True if the page exists (HTTP status 200), False otherwise
def webpage_exists(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

### Script that fetches all links to Bekanntmachungen from a given webpage
# NOTE: current fine-tuned for the BMBF website due to how pagination works there
# Input:
# - [str] base_url: the base URL of the website (e.g., "https://www.bmbf.de")
# - [str] search_url: the URL of the page listing Bekanntmachungen
# Output:
# - [str set] all_links: a list of unique URLs to Bekanntmachungen found on the page
def get_bekanntmachung_links(base_url, search_url):
    start = time()  # Start the timer to measure execution time
    print(f"Fetching Bekanntmachung links from {search_url}...")
    all_links = set() # Use a set to avoid duplicates

    # Identify the position of the page count in the search URL
    index = '1'
    current_search_url = search_url  # Start with the initial search URL

    # Loop through the pages to retrieve all links
    # While the page exists, continue fetching links
    while webpage_exists(current_search_url):

        try:
            # Update the search URL with the current page index
            print(f"Fetching Bekanntmachungen from page {index} on website {base_url}...")
            current_search_url = re.sub(r'D\d+&resultsPerPage=', f'D{index}&resultsPerPage=', search_url)
            index = str(int(index) + 1)  # Increment the page index

            # Find the links on the current search page
            response = requests.get(current_search_url)
            sleep(random.uniform(1,3))  # Sleep to avoid overwhelming the server with requests
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the container for search results (you'll need to inspect the page to find the correct class/ID)
            results_container = soup.find('table', class_='stack')  # Adjust the class name based on actual HTML structure

            if results_container:
                # Find all <a> tags within the results container that link to Bekanntmachungen
                links_tags = results_container.find_all('a', href=lambda href: href and 'SharedDocs/Bekanntmachungen/' in href)

                for link_tag in links_tags:
                    href = link_tag.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        all_links.add(full_url)

            else:
                print("Could not find the search results container. Check HTML structure.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the page: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    end = time()  # End the timer
    print(f"Finished fetching links in {end - start:.2f} seconds.")
    return list(all_links)



### Function to extract main content from the html page of a Bekanntmachung while preserving formatting
# Input:
# - [str] url: the URL of the Bekanntmachung page to extract content from
# Output:
# - [str] extracted text: the main text content of the Bekanntmachung, formatted with paragraphs, headings, and list items
def extract_main_text_via_url(url: str) -> str:

    print(f"Extracting content from {url}...")
    # Fetch the page content
    response = requests.get(url, timeout=15)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get the main content container.
    main_content_container = soup.find('div', id='content') # General content ID
    # main_content_container = soup.find('div', class_='document') # Common for structured documents
    # if not main_content_container:
    #     main_content_container = soup.find('div', id='content') # General content ID
    # if not main_content_container:
    #     main_content_container = soup.find('main') # HTML5 main content tag
    # if not main_content_container:
    #     main_content_container = soup.find('div', class_='richtext') # Another common class for text content
    
    if not main_content_container:
        # Fallback to body if no specific container is found.
        # This will include much more boilerplate (nav, footer, etc.) but ensures *some* text is captured.
        print(f"  Warning: Could not find a specific main content container for {url}. Extracting from <body>.")
        main_content_container = soup.find('body')

    if not main_content_container:
        return "Could not find any content to extract from this page."

    # Type guard to ensure we have a Tag object before calling find_all
    if not isinstance(main_content_container, BeautifulSoup.Tag):
        print(f"  Warning: Expected a Tag for main content but got {type(main_content_container)}. Extracting its text directly.")
        return main_content_container.get_text(strip=True)

    # Iterate over relevant block-level elements (paragraphs, headings, lists) to apply heuristics
    extracted_lines = []
    common_german_acronyms = sorted(["GmbH", "GbR", "kW", "mRNA", "WiFi", "UVLicht", "h2o", "H2O", "CO2", "co2", "IoT"], key=len, reverse=True) # List of acronyms that should not be split by spaces. Sorted from longest to shortest to ensure appropriate heuristics separation | NOTE: can cause issues with some acronyms if they are included in each other (e.g. O2 and CO2)
    common_file_extensions = ["php","txt","docx","csv","html","pdf","css","xml","json","doc","xls","xlsx", "ppt", "pptx", "odt", "ods", "odp", "jpg", "png", "gif", "svg", "bmp", "zip", "rar", "7z", "tar.gz", "mp4", "wav", "mp3", "avi", "mov"] # List of common file extensions
    # Look for common text-holding tags within the main content container
    for element in main_content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        text = element.get_text(strip=True) # Get text, remove leading/trailing whitespace
        # Remove unicode characters that can cause issues in text processing
        text = text.replace('\u2001', ' ').replace('\xad', '').replace('\xa0', ' ') # Replace non-breaking spaces and soft hyphens
        #text = re.sub(r'[\u200B-\u200F\uFEFF\u2060-\u206F\uFFF0-\uFFFF\uFFFE-\uFFFF]', '', text) # Remove zero-width characters and other invisible characters
        # text = ''.join(char for char in text if 
        #            (0x20 <= ord(char) <= 0x7E) or # Printable ASCII
        #            (ord(char) in [0x0A, 0x0D, 0x09]) or # Newline, Carriage Return, Tab
        #            (ord(char) >= 0x80)) # Allow all non-ASCII unicode (umlauts, etc.)
        # Use heuritics to address possible omissions of soft returns and missing spaces after semi-colons, while preserving URLs and e-mails
        text = re.sub(r'([A-Za-z])([\(])', r'\1 \2', text) # Add space between letter and opening parenthesis
        text = re.sub(r'([A-Za-z]|[:-])(http)', r'\1 \2', text) # Add space between letter or semi-colon and "http"
        text = re.sub(r'_ +', '_', text) # Remove multiple spaces after underscores that could break down URLs
        text = text.replace('\u2080', '0') # Replace subscript zero with normal zero for consistency
        text = text.replace('\u2081', '1') # Replace subscript one with normal one for consistency
        text = text.replace('\u2082', '2') # Replace subscript two with normal two for consistency
        text = text.replace('\u2083', '3') # Replace subscript three with normal three for consistency
        text = text.replace('\u2084', '4') # Replace subscript four with normal four for consistency
        text = text.replace('\u2085', '5') # Replace subscript five with normal five for consistency
        text = text.replace('\u2086', '6') # Replace subscript six with normal six for consistency
        text = text.replace('\u2087', '7') # Replace subscript seven with normal seven for consistency
        text = text.replace('\u2088', '8') # Replace subscript eight with normal eight for consistency
        text = text.replace('\u2089', '9') # Replace subscript nine with normal nine for consistency
        text = text.replace('\u00B2', '2') # Replace superscript two with normal two for consistency
        text = text.replace('\u00B3', '3') # Replace superscript three with normal three for consistency
        text = text.replace('\u00B9', '1') # Replace superscript one with normal one for consistency
        # Sometimes the original URLs can incorrectly contain extra spaces if they are between parentheses or brackets. They should be removed
        surrounded_broken_link_pattern = re.compile(r'(\(https?://[^\)]+\)|\[https?://[^\]]+\]|\(www\.[^\)]+\)|\[www\.[^\]]+\]|[\w\.-]+@[\w\.-]+\.\w+)')
        internal_url_space_pattern = re.compile(r'([a-zA-Z0-9\-_.~:%\.])\s+([a-zA-Z0-9\-_.~:%\.])')
        text = surrounded_broken_link_pattern.sub(lambda m: (internal_url_space_pattern.sub(r'\1\2', m.group(1)) if ' ' in m.group(1) else m.group(1)), text)
        # Processing URLs, e-mails and common exception terms
        url_pattern = r'(https?://[^\s]+|http?://[^\s]+|www\.[^\s]+|[\w\.-]+@[\w\.-]+\.\w+)'
        urls_found = re.findall(url_pattern, text)
        for i, url in enumerate(urls_found): # Replace each URL with a unique placeholder
            placeholder = f"__URL_{i}__"
            text = text.replace(url, placeholder)
        for i, term in enumerate(common_german_acronyms): # Replace each exception term with a unique placeholder
            placeholder2 = f"__EXC_{i}__"
            text = text.replace(term, placeholder2)
        # Heuristics that should not apply to URLs and exception terms
        text = re.sub(r'([.,:;!?\)])([A-Za-z]|(?<!\d\.)\d[-.\)])', r'\1 \2', text) # Add space after punctuation if followed by a letter, or number indicating a section or list item
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text) # Add space after number if followed by a letter
        text = re.sub(r'(\d+)\s*%', r'\1 %', text) # Ensure space before percentage sign (if not there) for consistency purposes
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # Add space between lowercase and uppercase letters
        text = re.sub(r'([A-Z][A-Z]+)([a-z])', r'\1 \2', text) # Add space between acronym and lowercase letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text) # Add space between letter followed by a number
        # Replace placeholders back with their original values
        for i, url in enumerate(urls_found):
            placeholder = f"__URL_{i}__"
            text = re.sub(
                rf'(\({placeholder}\))|(\[{placeholder}\])|(:)?{placeholder}|{placeholder}', 
                lambda m: (f' {url} ' if m.group(3) else (f': {url} ' if m.group(2) else url)), 
                text
                )
        for i, term in enumerate(common_german_acronyms):
            placeholder2 = f"__EXC_{i}__"
            text = text.replace(placeholder2, term)
        # Other heuristics to apply after links and acronyms have been reinstated
        text = re.sub(r'(?<![\[\(\s])(https?)', r' \1', text) # Add a space before http or https if it is not preceeded by a parenthesis or bracket and there is not already a space before it
        text = re.sub(r'(?<![/\[\(\s])(www)', r' \1', text) # Add a space before www if it is not preceeded by a parenthesis or bracket or / and there is not already a space before it
        text = re.sub(r'(?<!\s)(\[https?|\(https?)', r' \1', text) # Add a space before [http(s) or (http(s) if there is not already a space before it
        text = re.sub(r'((https?://|www\.)[^\s]+?)([.,;])(?=\s|$)', r'\1 \3', text) # Add a space between an URL and the following dot, comma or semi-colon, only if it is followed by a space or end of line
        text = re.sub(r'(\[(https?://|www\.)[^\s]+?\])([.,;])', r'\1 \3', text) # Add a space between an URL within brackets and the following dot, comma or semi-colon
        text = re.sub(r'(\((https?://|www\.)[^\s]+?\))([.,;])', r'\1 \3', text) # Add a space between an URL within parentheses and the following dot, comma or semi-colon
        # Insert a space after common file extensions if there is not already one, while taking care not to break any URL
        extensions_pattern = r"(?:" + "|".join(re.escape(ext) for ext in common_file_extensions) + r")"
        pattern = re.compile(r'(https?://[^\s]+?\.)'+r'('+extensions_pattern+r')'+r'(?![0-9\-_./?#&=])'+r'([a-zA-Zäöüß])')
        text = pattern.sub(r'\1\2 \3', text)
        # Add a space before and after an acronym (possibly between parentheses) if these are missing
        word_char = r'[a-zA-Z0-9]' # Characters for which a space should be added between it and the acronym
        for acr in common_german_acronyms:
            escaped_acronym = re.escape(acr) # The term with backslashes properly inserted for regular expresison processing
            acronym_pattern = rf"(?P<acronym>(?:\({escaped_acronym}\))|{escaped_acronym})" # Creates a capturing group called <acronym> for either acronym or (acronym) | TODO: this can cause incorrect segmentation if an acronym is contained in another (e.g. O2 in CO2). Check how to address this.
            text = re.sub(acronym_pattern + rf'(?={word_char})', r'\g<acronym> ', text) # Add a space between acronym and a subsequent word character if not already there
            text = re.sub(rf'(?<={word_char})' + acronym_pattern, r' \g<acronym>', text) # Add a space between a word character and subsequent acronym if not already there
        text = re.sub(r'\s+', ' ', text) # Change multiple spaces to a single one
        text = re.sub(r'([\[({])\s*', r'\1', text) # Remove spaces after opening bracket or parenthesis
        text = re.sub(r'\s*([\])}])', r'\1', text) # Remove space before closing brackets or parenthesis 
        
        if not text: # Skip empty elements
            continue

        if element.name == 'p':
            extracted_lines.append(text)
        elif element.name.startswith('h'): # Headings
            extracted_lines.append(f"\n{text}\n") # Add extra newlines for emphasis around headings
        elif element.name == 'li': # List items
            extracted_lines.append(f"- {text}") # Format as a list item
        elif element.name == 'div' and text:
            # For divs, if they contain substantial text and are not just containers,
            # treat them as a block. This might need fine-tuning.
            # Avoid adding too many empty lines if divs are nested.
            extracted_lines.append(text)

    # Join all collected lines with double newlines to simulate paragraph breaks
    # Filter out any empty strings that might have resulted from elements with no text
    return '\n\n'.join(filter(None, extracted_lines)).strip()


### Function to iterate over a list of URLs, extract the webpage contents, and save it to text files
# Input:
# - [str] url_file: path to a file containing URLs to scrape (one URL per line)
# - [str] output_dir: directory where the extracted content will be saved  
# - [int] min_delay_seconds: minimum delay between requests to avoid overwhelming the server
# - [int] max_delay_seconds: maximum delay between requests to avoid overwhelming the server   
# - [str] log_file: file where logs of failed URLs will be saved
# - [None|int] debug: if not set to None, process only the debug first examples
# Output:
# - None: the function saves the extracted content to text files in the specified output directory
def scrape_bekanntmachungen_content(
        url_file: str,
        output_dir: str, 
        min_delay_seconds: int = 1, 
        max_delay_seconds: int = 3, 
        log_file: str = './meta_data/failed_content_urls.txt', 
        debug: Optional[int] = None) -> None:

    start = time()  # Start the timer to measure execution time

    try:
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: URL file '{url_file}' not found. Please ensure it exists.")
        return
    
    # If debug is not None, process only the first debug URLs for testing
    if debug is not None:
        urls = urls[:debug]  # For testing, limit to first debug URLs

    print(f"Found {len(urls)} URLs to process.")

    # Empty the output directory if it already contains files
    output_file_list = os.listdir(output_dir)
    if len(output_file_list)!=0:
        print(f'Output directory {output_dir} non-empty: removing existing {len(output_file_list)} file(s)')
        for idx in range(len(output_file_list)):
            os.remove(os.path.join(output_dir,output_file_list[idx]))
    
    successful_count = 0
    failed_urls = []

    for i, url in enumerate(urls):
        print(f"Processing URL {i+1}/{len(urls)}: {url}")
        
        # Create a safe and unique filename for each URL's content
        # Using parts of the URL path and query parameters can help make it unique
        parsed_url = urlparse(url)
        # Example: use the last path segment (e.g., "2025-02-24-bekanntmachung-egaroh-junior")
        # and sanitize it to be a valid filename.
        file_name_segment = parsed_url.path.split('/')[-1].replace('.html', '').replace('.', '_').replace('-', '_')
        
        # Add a fallback for very short or generic URLs
        if not file_name_segment or len(file_name_segment) < 10: 
            # If the segment is too short or empty, use a hash or index
            file_name_segment = f"page_{i+1}_{hash(url) % 10000}" # Simple hash for uniqueness

        output_filepath = os.path.join(output_dir, f"{file_name_segment}.txt")

        # Skip if the file already exists (useful for resuming interrupted scrapes)
        if os.path.exists(output_filepath):
            print(f"  Skipping: Content already exists at {output_filepath}")
            successful_count += 1
            continue

        try:
            
            # Pass the original URL to the extraction function for better error messages
            main_text = extract_main_text_via_url(url)

            if main_text and main_text.strip() != "Could not find any content to extract from this page.":
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    outfile.write(main_text)
                print(f"  Successfully saved content to {output_filepath}")
                successful_count += 1
            else:
                print(f"  Could not extract meaningful text from {url}. Page might have an unexpected structure or no content.")
                failed_urls.append(url)

        except requests.exceptions.HTTPError as e:
            print(f"  HTTP Error {e.response.status_code} for {url}: {e}")
            failed_urls.append(url)
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection Error for {url}: {e}")
            failed_urls.append(url)
        except requests.exceptions.Timeout:
            print(f"  Timeout Error for {url}")
            failed_urls.append(url)
        except requests.exceptions.RequestException as e:
            print(f"  General Request Error for {url}: {e}")
            failed_urls.append(url)
        except Exception as e:
            print(f"  An unexpected error occurred for {url}: {e}")
            failed_urls.append(url)

        # Introduce a random delay to be polite and avoid rate limits
        sleep_duration = random.uniform(min_delay_seconds, max_delay_seconds)
        sleep(sleep_duration)

    # Summary of results
    print(f"Successfully processed {successful_count} out of {len(urls)} URLs.")
    if failed_urls:
        print(f"Failed to process {len(failed_urls)} URLs:")
        for failed_url in failed_urls:
            print(f"  - {failed_url}")
        # Optionally save failed URLs to a file for later review
        with open(log_file, 'w', encoding='utf-8') as f:
            for fu in failed_urls:
                f.write(fu + '\n')
        print(f"Failed URLs saved to {log_file}")

    end = time()  # End the timer
    print(f"Total execution time: {end - start:.2f} seconds.")


# =======================================================
# === Building the dataset for supervised fine-tuning ===
# =======================================================

### Function that builds the training set for supervised fine-tuning of the LLM
# Input:
# - [str] doc_folder_path: path to the folder containing the source documents (currently either PDF or text)
# - [str] ground_truth_filepath: path to the json file containing the ground truth information
# - [str] output_filepath: path where the resulting json file will be saved.
# - [str] embedding_model_name: name of the embedding model to be used for the information extraction
# - [int] chunk_size: size of the text chunks to be created from the document
# - [int] chunk_overlap: overlap between the text chunks 
# - [int] top_k_retrieval: number of top relevant chunks to retrieve for each query
# Output:
# - None
def generate_training_dataset_prompts(
    doc_folder_path: str,
    ground_truth_filepath: str,
    output_filepath: str,
    prompts_filepath: str,
    embedding_model_name: str = "models/embedding-001",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    #max_context_length: int = 4000,  # Max context length for the prompt
    top_k_retrieval: int = 4  # Number of chunks to retrieve, similar to default in RetrievalQA
) -> None:
    """
    Generates a JSON file containing prompts for training a model,
    mimicking the RAG prompt construction of the grant summarisation agent.

    Each entry in the output JSON will have the format:
    {
        "file_name": "path/to/document.txt",
        "field": "objective",
        "prompt": "Full prompt text including retrieved context and query",
        "answer": "Answer retrieved from the ground truth files"
    }

    Args:
        doc_paths (List[str]): A list of absolute paths to the document files (PDF or text).
        output_filepath (str): The absolute path to the JSON file where the dataset will be saved.
        prompts_filepath (str): The absolute path to the JSON file where the prompts are stored.
        embedding_model_name (str): Name of the embedding model to use (e.g., "models/embedding-001").
        chunk_size (int): Size of the text chunks for document splitting.
        chunk_overlap (int): Overlap between text chunks.
        max_context_length (int): Maximum character length for the context included in the prompt.
                                  This simulates truncation if the retrieved context is too long.
        top_k_retrieval (int): Number of top relevant chunks to retrieve for each query.
    """
    start_time = time()
    doc_paths = [os.path.join(doc_folder_path, doc_name) for doc_name in os.listdir(doc_folder_path) if doc_name.endswith('.txt') or doc_name.endswith('.pdf')]
    training_data_entries = []

    # Initialise embedding model and splitter
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY not found among the environment variables defined in .env")
    os.environ['GOOGLE_API_KEY'] = gemini_api_key

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load the base prompts for each field
    with open(prompts_filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # queries = {
    #     "objective": objective_query,
    #     "inclusion_criteria": inclusion_query,
    #     "exclusion_criteria": exclusion_query,
    #     "deadline": deadline_query,
    #     "max_funding": max_funding_query,
    #     "max_duration": max_duration_query,
    #     "procedure": procedure_query,
    #     "contact": contact_query,
    #     "misc": misc_query,
    # }

    # List the ground truth files
    ground_truth_file_list = [e for e in os.listdir(ground_truth_filepath) if ".json" in e]

    for doc_path in doc_paths:
        print(f"Processing document: {doc_path}")

        # Load the document
        if doc_path.endswith('.pdf'):
            loader = PyPDFLoader(doc_path)
        elif doc_path.endswith('.txt'):
            loader = TextLoader(doc_path, encoding='utf-8')
        else:
            print(f"Skipping {doc_path}: Unsupported file type.")
            continue

        # Retrieve and load the corresponding ground truth file
        base_name = os.path.basename(doc_path).rsplit('.', 1)[0]
        ground_truth_file = [e for e in ground_truth_file_list if base_name in e][0]
        if ground_truth_file:
            with open(os.path.join(ground_truth_filepath, ground_truth_file), 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        else:
            print(f"  Warning: No ground truth file found for {doc_path}.")
            ground_truth = {}

        try:
            documents = loader.load()
            split_docs = text_splitter.split_documents(documents)
            if not split_docs:
                print(f"No content extracted from {doc_path}. Skipping.")
                continue

            index = DocArrayInMemorySearch.from_documents(split_docs, embeddings)
            retriever = index.as_retriever(search_kwargs={"k": top_k_retrieval})

            for field, query_string in queries.items():
                # Retrieve relevant chunks
                retrieved_docs = retriever.invoke(query_string)
                relevant_parts = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # # Apply context truncation logic, similar to evaluate_agent.py
                # NOTE: removed for the dataset generation to keep as much context as possible
                # disclaimer = '... [context truncated due to length]'
                # if len(relevant_parts) > max_context_length - len(disclaimer):
                #     relevant_parts = relevant_parts[:max_context_length - len(disclaimer)] + disclaimer

                # Construct the full prompt as it would be sent to the LLM by RetrievalQA
                # This template is a common default for RetrievalQA with chain_type="stuff".
                full_prompt_template = (
                    "Use the following pieces of context to answer the user's question.\n"
                    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
                    "----------------\n"
                    "{context}\n"
                    "----------------\n"
                    "Question: {question}\n"
                )
                final_prompt = full_prompt_template.format(context=relevant_parts, question=query_string)
                corresponding_ground_truth = ground_truth.get(field, "")

                training_data_entries.append({
                    "file_name": doc_path,
                    "field": field,
                    "prompt": final_prompt,
                    "answer": corresponding_ground_truth 
                })
        except Exception as e:
            print(f"An error occurred while processing {doc_path}: {e}")
            continue

    # Save the generated dataset to a JSON file
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(training_data_entries, f, indent=4, ensure_ascii=False)

    end_time = time()
    print(f"Generated training dataset prompts in {end_time - start_time:.2f} seconds.")
    print(f"Saved {len(training_data_entries)} entries to {output_filepath}")


### Function that splits the saved dataset into training and testing sets and converts them into the proper HuggingFace data format
# Input:
# - [str] database_path: path to the json file containing the data
# - [str] training_set_path: path to where to save the CSV file containing the training set
# - [str] testing_set_path: path to where to save the CSV file containing the testing set
# - [float] train_proportion: percentage of the samples assigend to the training set
# - [int] random_seed: random seed for random splitting between the training and testing sets
# Output:
# - None
def split_train_test(database_path: str, training_set_path: str, testing_set_path: str, train_proportion: float = 0.7, random_seed: int = 42):

    # Load the database
    with open(database_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Retrieve the total list of original files
    filename_list = list(set([e["file_name"] for e in data]))

    # Randomly split the files between training and testing sets 
    random.seed(random_seed)
    random.shuffle(filename_list)
    train_size = int(len(filename_list) * train_proportion)
    train_files = filename_list[:train_size]
    test_files = filename_list[train_size:]
    train_samples = [e for e in data if e["file_name"] in train_files]
    test_samples = [e for e in data if e["file_name"] in test_files]

    # Format both training and testing sets according to the HuggingFace expected formats
    # The samples should be in a CSV file with columns ["file_path", "field", "text"]
    # The "text" field should follow the format "human: sample['prompt'] \n bot: sample['answer']"
    train_frame = pd.DataFrame(columns=["file_path", "field", "text"])
    for idx, sample in enumerate(train_samples):
        new_text = f"human: {sample['prompt']} \n bot: {sample['answer']}"
        train_frame.loc[idx] = {"file_path": sample["file_name"], "field": sample["field"], "text": new_text}
    test_frame = pd.DataFrame(columns=["file_path", "field", "text"])
    for idx, sample in enumerate(test_samples):
        new_text = f"human: {sample['prompt']} \n bot: {sample['answer']}"
        test_frame.loc[idx] = {"file_path": sample["file_name"], "field": sample["field"], "text": new_text}

    # Save both training and testing sets in CSV formats
    train_frame.to_csv(training_set_path, index=False)
    test_frame.to_csv(testing_set_path, index=False)


# ======================================================
# === Building the structured database for Graph RAG ===
# ======================================================

### Helper function that given a path to a text file, returns a dictionary of all sections contained in the document.
# Input:
# - [str] doc_path: path to the text document containing the Bekanntmachung
# Output:
# - [Dict[str, Dict[str, str]]]: A dictionary where keys are section IDs and values are dictionaries containing 'text' and 'title'.
def extract_all_sections_from_document(doc_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parses a grant document and extracts all sections into a dictionary.

    The function identifies sections by numbers (e.g., "1.", "2.1", "7.2.1")
    at the beginning of a line. It also handles a special "introduction" section
    (text before section 1) and annexes (Anlage).

    Sections that only contain a title line and are immediately followed by
    subsections will have their title prepended to the first subsection's content.
    Sections that are empty (only title, no body, and no subsections) are discarded.

    Args:
        doc_path: The path to the text document containing the German call.

    Returns:
        A dictionary where keys are section IDs (e.g., "1.1") and values are
        dictionaries with 'text' (the full content) and 'title' (the title line)
        for that section.
        Returns an empty dictionary if the file is not found.
    """
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Error: Document not found at {doc_path}")
        return {}

    # Regex to find section numbers like 1., 1.1, 2.3.4 etc. at the start of a line.
    section_pattern = re.compile(r'^\s*(\d(?:\.\d+)*)\.?\s+.*')
    # Regex for Annexes (Anlage)
    annex_header_pattern = re.compile(r'^\s*Anlage\s*$', re.IGNORECASE)  # Matches lines with only "Anlage"

    raw_sections_data = {} # Stores {'section_id': {'title_line': '...', 'body': '...'}}
    
    current_section_id = 'introduction'
    current_title_line = '' # For 'introduction', this will remain empty
    current_content_body = []
    inside_annex = False
    annex_counter = 0

    lines = full_text.split('\n')

    for line_idx, line in enumerate(lines):
        section_match = section_pattern.match(line)
        is_annex_header = annex_header_pattern.match(line)

        if section_match or is_annex_header:
            # Before starting a new section, store the content of the previous one
            if current_section_id:
                raw_sections_data[current_section_id] = {
                    'title_line': current_title_line,
                    'body': '\n'.join(current_content_body).strip()
                }

            # Reset for the new section
            current_content_body = []
            current_title_line = line.strip() # Store the actual title line

            if is_annex_header:
                inside_annex = True
                annex_counter += 1
                current_section_id = f'annex {annex_counter}'
            elif section_match:
                new_section_number = section_match.group(1)
                if inside_annex:
                    current_section_id = f'annex {annex_counter}.{new_section_number}'
                else:
                    current_section_id = new_section_number
        else:
            # Append to the current section's body content
            current_content_body.append(line)

    # Store the last section after the loop finishes
    if current_section_id:
        raw_sections_data[current_section_id] = {
            'title_line': current_title_line,
            'body': '\n'.join(current_content_body).strip()
        }

    return raw_sections_data

### Helper function that cleans the extracted text chunks by applying text cleaning heuristics
# Input:
# - [str] text: the text chunk to be cleaned
# - [bool] add_tags: if True, add html style tags to indicate relevant information such as emails, phone numbers, amounts, dates
# Output:
# - [str]: the cleaned text chunk
def clean_extracted_text(text: str, add_tags: bool = False) -> str:
    # Add a space inside a word where one of the inside letters is upper-case (e.g. "undFrau" -> "und Frau")
    text = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', text)

    # Define common TLDs to make email regex more specific and avoid false positives.
    common_tlds = [
        # Generic
        'com', 'org', 'net', 'edu', 'gov', 'info',
        # EU Countries
        'at', 'be', 'bg', 'cy', 'cz', 'de', 'dk', 'ee', 'es', 'eu', 'fi', 'fr', 
        'gr', 'hr', 'hu', 'ie', 'it', 'lt', 'lu', 'lv', 'mt', 'nl', 'pl', 'pt', 
        'ro', 'se', 'si', 'sk',
        # Other specified countries
        'ca', 'ch', 'cn', 'jp', 'kr', 'mx', 'tw', 'uk', 'us', 'za'
    ]
    # Sort for consistency
    common_tlds.sort()
    tld_pattern_part = r'(?:' + '|'.join(re.escape(tld) for tld in common_tlds) + r')'

    # Add a space after an email address if it is immediately followed by a letter (e.g., "name@domain.deund" -> "name@domain.de und")
    email_spacing_pattern = rf'(\b[\w\.-]+@[\w\.-]+\.{tld_pattern_part})(\w)'
    text = re.sub(email_spacing_pattern, r'\1 \2', text, flags=re.IGNORECASE)

    # Add tags if enabled
    if add_tags:
        # Tag email addresses with <email> ... </email>
        email_pattern = rf'(\b[\w\.-]+@[\w\.-]+\.{tld_pattern_part}\b)'
        text = re.sub(email_pattern, r'<email>\1</email>', text)

        # Tag phone numbers with <phone> ... </phone> (e.g. 03 86/78 10-57 64, +49-30-123456, (030) 1234567, etc.)
        # Make sure that if one parenthesis is present, the other is also present
        phone_pattern = r'(Telefon\s*:)\s*(\+?[\d\s\-/\(]*[\d\s\-/\)]*\d)'
        text = re.sub(phone_pattern, r'\1 <phone>\2</phone>', text)
        # # Because the above pattern captures spaces for the phone number, we make sure that there are no leading or trailing spaces within the tags
        # text = re.sub(r'<phone>\s+([\d\+\s\-/\(\)]+)\s+</phone>', r'<phone>\1</phone>', text)

        # Tag amounts with <amount> ... </amount> (e.g. 1.000,00 €, 5000 EUR, 500 000 Euro, 10 Millionen EUR, etc.)
        amount_pattern = r'(\b\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?\s*(?:€|EUR|Euro|CHF)\b)'
        text = re.sub(amount_pattern, r'<amount>\1</amount>', text)
        amount_pattern2 = r'(\b\d{1,3}([,.]\d+)?\s*(Million|Millionen|Milliarde|Milliarden)\s*(?:€|EUR|Euro|CHF)\b)'
        text = re.sub(amount_pattern2, r'<amount>\1</amount>', text)

        # Tag dates with <date> ... </date> (e.g. 15.05.2006, 13. Januar 2010, etc.)
        date_pattern = r'(\b\d{1,2}\.\s?\d{1,2}\.\s?\d{2,4}\b|\b\d{1,2}\.?\s(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d{2,4}\b)'
        text = re.sub(date_pattern, r'<date>\1</date>', text)

        # Tag percentages with <percentage> ... </percentage> (e.g. 15 %, 7,5%, 2.3%, etc.)
        percentage_pattern = r'(\d+\s*%)'
        text = re.sub(percentage_pattern, r'<percentage>\1</percentage>', text)

        # Tag durations with <duration> ... </duration> (e.g. 6 Monate, 2 Jahre, etc.)
        # Also includes modifiers such as "bis ... zu", "maximal", "mindestens", "zwischen ... und"
        # Must exclude ages (e.g. "von Kinder unter 3 Jahren", "nicht älter als 65 Jahre")
        duration_pattern = r'((?:bis zu\s*|maximal\s*|zwischen \d+ und\s*|(?:von )?\d+ bis\s*|mindestens\s*|höchstens\s*)?\d+\s+(?:Monat(?:e|en)?|Jahr(?:e|en)?|Woche(?:n)?|Tag(?:e|en)?)\b)'
        text = re.sub(duration_pattern, r'<duration>\1</duration>', text)
        duration_pattern2 = r'(?:(?<=von Kinder unter )|(?<=nicht älter als )|(?<=nicht aelter als ))(<duration>)'+duration_pattern+r'(</duration>)'
        text = re.sub(duration_pattern2, r'\2', text)

        # Tag persons with <person> ... </person> (e.g. Dr. Max Mustermann, Prof. Dr. Erika Musterfrau, Frau Erika Musterfrau, etc.)
        # Titles (gender, professor or doctor) are optional, but at least one must be present to avoid false positives
        # Name detection is achieved by a regular expression of type (AB?C?|A?BC?|AB?C) where A, B, C are the title patterns
        title_pattern = r'(?:(?:Herr|Frau|Herrn|Mr\.|Ms\.)\s*)'
        professor_pattern = r'(?:Prof\.\s*|Professor(?:in)?\s*)'
        doctor_pattern = r'(?:Dr\.\s*(?:-Ing\.|-ing\.|rer\.\s*nat\.)?\s*)'
        firstname_pattern = r'(?:[A-ZÄÖÜ][a-zäöüß]+)?(?:(?:-|\s)[A-ZÄÖÜ][a-zäöüß]+)?\s*'
        lastname_pattern = r'[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?'
        person_pattern = r'(\b(?:' + title_pattern + professor_pattern + r'?' + doctor_pattern + r'?|' + title_pattern + r'?' + professor_pattern + doctor_pattern + r'?|' + title_pattern + r'?' + professor_pattern + r'?'+ doctor_pattern + r')' + firstname_pattern + r'?' + lastname_pattern + r'\b)'
        text = re.sub(person_pattern, r'<person>\1</person>', text)
        # The person tagging may have caught some extra words that are not names (e.g. "Telefon", "E-Mail", etc.), so we remove those from within the tags
        exclusion_pattern = r'(?:\s*)(Telefon|Beratungstelefon|Telefonnummer|E-Mail|Email|Mail|Fax|Telefax|Adresse|Anschrift|Straße|Str|Platz|Weg|Hausnummer|Nr|Postleitzahl|Ort|Stadt|Bundesland|Land|Webseite|Website|Internet)'
        text = re.sub(r'<person>\s*' + person_pattern + exclusion_pattern + r'\s*</person>', r'<person>\1</person> \2', text)
        
        # Make sure than any closing tag is followed by a space if followed by a letter
        text = re.sub(r'(</(email|phone|amount|date|percentage|duration|person)>)\s*(\w)', r'\1 \3', text)
        # Make sure that there are no closing or leading spaces within the tags
        text = re.sub(r'<(email|phone|amount|date|percentage|duration|person)>\s+([^<]+?)\s+</\1>', r'<\1>\2</\1>', text)

    return text


### Function that creates a structured database for the Graph RAG and saves it in jsonl format
# Input:
# - [str] data_folder_path: path to the folder containing the original Bekanntmachungen files in text format
# - [int] chunk_size: maximum size in characters allowed per chunk
# - [int] chunk_overlap: overlap between chunks in characters
# - [str] output_path: path to the jsonl file where the database should be saved
# - [bool] clean_text: if True, clean the text chunks by applying text cleaning heuristics
# - [bool] add_tags: if True, add html style tags to indicate relevant information such as emails, phone numbers, amounts, dates
# Output:
# - None
def create_structured_database_from_bekanntmachungen(data_folder_path: str, chunk_size: int, chunk_overlap: int, output_path: str, clean_text: bool=True, add_tags: bool=False) -> None:

    print(f'Creating the structured database from {data_folder_path} with chunk size {chunk_size} and overlap {chunk_overlap} for Graph RAG ...\n')
    if clean_text:
        print('Text cleaning enabled: applying text cleaning heuristics to each chunk.\n')
    if add_tags:
        print('Tagging enabled: adding html style tags to indicate relevant information such as emails, phone numbers, amounts, dates.\n')

    # List the files in the original folder
    file_list = [e for e in os.listdir(data_folder_path) if e.endswith('.txt')]
    file_paths = [os.path.join(data_folder_path, e) for e in file_list]

    # Initialise the Langchain text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

    # Loop on the files 
    for idx, file_path in enumerate(file_paths):
        # Initialise the chunk index
        chunk_idx = 1

        print(f'Processing file {idx+1}/{len(file_paths)}: {file_path}')

        # Check if the document is a Bekanntmachung or not
        file_name = os.path.basename(file_path)
        if "bekanntmachung" in file_name.lower() and not "aenderung" in file_name.lower() and not "vergabe" in file_name.lower():
            document_type = "bekanntmachung"
        elif "aenderung" in file_name.lower():
            document_type = "modification"
        elif "vergabe" in file_name.lower():
            document_type = "grant"
        else:
            document_type = "other"

        # Create meta-data dictionary for the document
        meta_data = {
            "document_id": file_name,
            "document_path": file_path,
            "document_type": document_type,
        }

        # Extract text sections from document
        if document_type == "bekanntmachung":
            sections = extract_all_sections_from_document(file_path)
            # Split each section into chunks
            for section_id, section_data in sections.items():
                if section_data['body']:
                    text = section_data['body']
                    # Build the hierarchical section title string
                    if section_id == 'introduction':
                        meta_data["section_title"] = 'introduction'
                    else:
                        parent_parts = section_id.split('.')
                        title_hierarchy = []
                        for i in range(len(parent_parts)):
                            parent_id = '.'.join(parent_parts[:i+1])
                            if parent_id in sections and sections[parent_id]['title_line']:
                                title_hierarchy.append(sections[parent_id]['title_line'])
                        meta_data["section_title"] = "; ".join(title_hierarchy)
                    meta_data["section_id"] = section_id
                    # The section level is the number of dots contained in the section ID 
                    meta_data["section_level"] = f"{section_id.count('.') + 1}"
                    chunks = text_splitter.split_documents([Document(page_content=text, metadata=meta_data)])

                    with open(output_path, 'a', encoding='utf-8') as f:
                        for i, chunk in enumerate(chunks):
                            chunk.metadata["chunk_id"] = f"{chunk_idx}"
                            unique_id = f"{meta_data['document_id'].rsplit('.', 1)[0]}_sec-{section_id}_chunk-{chunk_idx}"
                            chunk_idx += 1  
                            chunk.metadata["id"] = unique_id    
                            # Apply text cleaning and tag addition if enabled
                            if clean_text:
                                chunk.page_content = clean_extracted_text(chunk.page_content, add_tags=add_tags)   
                            # Create the final dictionary structure for the JSONL line
                            record = {
                                "text": chunk.page_content,
                                "metadata": chunk.metadata
                            }
                            
                            # Write the JSON object followed by a newline character
                            json_line = json.dumps(record, ensure_ascii=False)
                            f.write(json_line + '\n')

        else: # The document is not a Bekanntmachung and is usually much shorter, so the full document is used as chunk instead
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if clean_text:
                text = clean_extracted_text(text, add_tags=add_tags)
            meta_data["section_id"] = "document"
            meta_data["section_level"] = "1"
            meta_data["section_title"] = "full_document"
            meta_data["chunk_id"] = "1"
            unique_id = f"{meta_data['document_id'].rsplit('.', 1)[0]}_sec-document_chunk-1"  
            meta_data["id"] = unique_id     
            record = {
                "text": text,
                "metadata": meta_data
            }
            with open(output_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + '\n')


### Function that extracts the entities and relationships from the structured database.
### Entities and relationships are represented as triplets of (subject, predicate, object).
### Extracts structural triplets (document, HAS_SECTION, section) by parsing the database file.
### Extracts conceptual triplets (e.g. (document, DEFINES_OBJECTIVE, objective)) by a LLM call.
### Save all extracted concepts into a jsonl file with fields ["subject", "predicate", "object", "source_document_id", "source_section_title"].
# Input:
# - [str] database_path: path to the jsonl file containing the structured database of chunks
# - [str] output_path: path to the jsonl file where the database of entities and relationships should be saved.
# - [str] prompt_filepath: path to the file containing the LLM prompt for concept extraction.
# - [str] llm_name: name of the LLM to be used for triplet extraction.
# - [float] temperature: temperature parameter of the LLM to be used for entity extraction
# Output:
# - None
def extract_entities_and_relationships(
        database_path: str,
        output_path: str, 
        prompt_filepath: str, 
        llm_name: str='gemini-2.5-flash',
        temperature: float=0.1):
    
    start_time = time()
    print('Extracting entities and relationships from the structured database ...')

    # Load the jsonl file containing the chunks
    with open(database_path, 'r', encoding='utf-8') as f:
        chunks_database = f.readlines()

    # --- Intelligent Resuming Logic ---
    # Load existing triplets to avoid duplicates if the script is resumed.
    existing_triplets = set()
    if os.path.exists(output_path):
        print(f"Output file found at {output_path}. Loading existing triplets to prevent duplicates.")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    triplet = json.loads(line)
                    # Create a unique, hashable representation of the triplet
                    if all(k in triplet for k in ["subject_type", "subject_value", "predicate", "object_type", "object_value"]):
                        existing_triplets.add((triplet["subject_type"], triplet["subject_value"], triplet["predicate"], triplet["object_type"], triplet["object_value"]))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line in existing output file: {line.strip()}")
        print(f"Loaded {len(existing_triplets)} unique existing triplets.")

    # Create the document structure triplets by iterating over the chunks
    document_structure_triplets = []
    print('Creating the document structure triplets ...')
    for chunk in chunks_database:
        chunk_data = json.loads(chunk)
        # Retrieve the title of the section at the deepest level only
        titles = chunk_data["metadata"]["section_title"].split("; ")
        deepest_title = titles[-1] if titles else ""

        new_triplet = {
            "subject_type": "DOCUMENT",
            "subject_value": chunk_data["metadata"]["document_id"],
            "predicate": "HAS_SECTION",
            "object_type": "SECTION",
            "object_value": deepest_title,
            "source_document_id": chunk_data["metadata"]["document_id"],
            "source_section_title": deepest_title,
            "chunk_id": chunk_data["metadata"]["chunk_id"]
        }

        # Add triplet only if it's not already in the existing set
        triplet_key = (new_triplet["subject_type"], new_triplet["subject_value"], new_triplet["predicate"], new_triplet["object_type"], new_triplet["object_value"])
        if triplet_key not in existing_triplets:
            document_structure_triplets.append(new_triplet)
            existing_triplets.add(triplet_key) # Also add to the in-memory set to handle duplicates within the same run

    # Save the document structure triplets in the output jsonl file
    with open(output_path, 'a', encoding='utf-8') as f:
        for triplet in document_structure_triplets:
            json_line = json.dumps(triplet, ensure_ascii=False)
            f.write(json_line + '\n')

    end_structure = time()
    print(f"Document structure triplets completed in {end_structure - start_time:.2f} seconds.")
    
    # --- Conceptual Triplet Extraction ---
    # Reload existing triplets to include structural ones and prepare for conceptual triplet resume logic.
    # We create a set of (doc_id, chunk_id) tuples for which conceptual triplets have already been extracted.
    processed_chunks = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    triplet = json.loads(line)
                    # A conceptual triplet is any triplet that is not structural.
                    if triplet.get("predicate") != "HAS_SECTION" and "source_document_id" in triplet and "chunk_id" in triplet:
                        processed_chunks.add((triplet["source_document_id"], triplet["chunk_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue # Ignore malformed lines or lines without the required keys
    # Extracts the conceptual triplets by calling the LLM
    start_concept = time()
    print(f'Extracting the conceptual triplets by querying {llm_name} ...')
    # Initialise the LLM
    load_dotenv()
    graph_rag_api_key = os.getenv("GRAPH_RAG_GEMINI_API_KEY")
    if graph_rag_api_key is None:
        raise ValueError("GRAPH_RAG_GEMINI_API_KEY not found among the environment variables defined in .env")

    llm = ChatGoogleGenerativeAI(model=llm_name, temperature=temperature, google_api_key=graph_rag_api_key)

    # Load the prompt template from the provided json file
    with open(prompt_filepath,'r',encoding='utf-8') as f:
        prompt_template_string = json.load(f)['prompt_template']

    prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
    # Define the Pydantic model for the expected output
    class Triplet(BaseModel):
        """A structured representation of a relationship triplet."""
        subject_type: str = Field(description="The type of the subject that takes values among the previously defined entities.")
        subject_value: str = Field(description="The value taken by the subject, of maximum 3 sentences or 150 words.")
        predicate: str = Field(description="The predicate or type of the relationship.")
        object_type: str = Field(description="The type of the object that takes values among the previously defined entities.")
        object_value: str = Field(description="The value taken by the object, of maximum 3 sentences or 150 words.")

    # Define the chain
    parser = PydanticOutputParser(pydantic_object=Triplet)
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    chain = (prompt_template | llm | output_fixing_parser)

    # Loop on the chunks of the structured database
    for i, chunk in enumerate(chunks_database):
        chunk_data = json.loads(chunk)
        source_document_id = chunk_data["metadata"]["document_id"]
        chunk_id = chunk_data["metadata"]["chunk_id"]

        # --- Intelligent Resuming for Conceptual Triplets ---
        if (source_document_id, chunk_id) in processed_chunks:
            print(f"Skipping chunk {i+1}/{len(chunks_database)}: Conceptual triplets for doc '{source_document_id}' chunk '{chunk_id}' already exist.")
            continue

        print(f"Processing chunk {i+1}/{len(chunks_database)}: doc '{source_document_id}' chunk '{chunk_id}'")

        # Retrieve the title of the section at the deepest level only
        titles = chunk_data["metadata"]["section_title"].split("; ")
        deepest_title = titles[-1] if titles else ""

        # Prompt input
        prompt_input = {
            "document_title": chunk_data["metadata"]["document_id"],
            "section_title": chunk_data["metadata"]["section_title"],
            "text": chunk_data["text"],
            "format_instructions": parser.get_format_instructions(),
        }
        try:
            # The parser returns a Pydantic object. It might be None if the LLM returns nothing.
            result_triplet = chain.invoke(prompt_input)
            
            if result_triplet and all([result_triplet.subject_type, result_triplet.subject_value, result_triplet.predicate, result_triplet.object_type, result_triplet.object_value]):
                new_triplet = result_triplet.dict() # Convert Pydantic model to dictionary
                new_triplet["source_document_id"] = source_document_id
                new_triplet["source_section_title"] = deepest_title
                new_triplet["chunk_id"] = chunk_id
                # Save the new triplet in the output jsonl file
                with open(output_path, 'a', encoding='utf-8') as f:
                    json_line = json.dumps(new_triplet, ensure_ascii=False)
                    f.write(json_line + '\n')
            else:
                print(f"  - No conceptual triplet found for chunk {chunk_id} of document {source_document_id}.")
        except ResourceExhausted as e:
            print(f"  - Resource exhausted for chunk {chunk_id}. Waiting 60 seconds before retrying. Error: {e}")
            sleep(60)
            # You might want to re-add the chunk to a queue to retry later
        except Exception as e:
            print(f"  - Error generating conceptual triplet for chunk {chunk_id} of document {source_document_id}: {e}. Skipping.")
    end_concept = time()
    print(f"Conceptual triplets completed in {end_concept - start_concept:.2f} seconds.")
    print(f"Entities and relationships extracted and saved in {output_path} in {end_structure - start_time:.2f} seconds.")


# ======================
# === Main function ===
# ======================
if __name__ == "__main__":

    ### Load the YAML config file and main logic
    try:
        config = load_config_from_yaml(r'./config.yaml')
    except:
        print("ERROR: YAML config file './config.yaml' not found!")
        sys.exit(1)

    run_steps = config.get('run_steps', {})

    ### Build the database of Bekanntmachungen
    if run_steps.get('build_database', False):
        print("\n--- Building the Bekanntmachungen database ---")
        parameters = config.get('build_database',{})

        # Building the list of Bekanntmachung links
        output_dir = parameters.get('output_dir', './data/')
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Get the links
        bekanntmachung_urls = get_bekanntmachung_links(parameters.get('bmbf_base_url'), parameters.get('bmbf_search_url'))
        print(f"Found {len(bekanntmachung_urls)} unique Bekanntmachung links.")
        # Save the list of links to a text file
        url_file = parameters.get('url_file')
        with open(url_file, 'w', encoding='utf-8') as f:
            for url in bekanntmachung_urls:
                f.write(url + '\n')

        # Retrive the contents of each Bekanntmachung and save them in text format
        scrape_bekanntmachungen_content(
            url_file, 
            output_dir, 
            min_delay_seconds=parameters.get('min_delay_seconds'), 
            max_delay_seconds=parameters.get('max_delay_seconds'), 
            log_file=parameters.get('log_file')
        )

        # NOTE: Debugging purposes
        # log_file = './meta_data/failed_content_urls.txt' # File where logs of failed URLs will be saved
        # log_file2 = './meta_data/failed_content_urls2.txt' # File where logs of failed URLs will be saved (when running again the script onto the log files)
        #scrape_bekanntmachungen_content(url_file, output_dir='./data_debug/', min_delay_seconds=min_delay_seconds, max_delay_seconds=max_delay_seconds, log_file=log_file, debug=20)
        # #Retry failed URLs from the log file
        #scrape_bekanntmachungen_content(log_file, output_dir, min_delay_seconds=min_delay_seconds, max_delay_seconds=max_delay_seconds, log_file=log_file2)

    ### Generate the ground truth database for supervised fine-tuning
    if run_steps.get('build_sft_dataset', False):
        print("\n--- Generating training dataset prompts ---")
        parameters = config.get('build_sft_dataset',{})
        
        generate_training_dataset_prompts(
            doc_folder_path=parameters.get('data_folder'),
            ground_truth_filepath=parameters.get('ground_truth_filepath'),
            output_filepath=parameters.get('output_filepath'),
            prompts_filepath=parameters.get('prompts_filepath'),
            embedding_model_name=parameters.get('embedding_model_name','models/embedding-001'),
            chunk_size=parameters.get('chunk_size',4000),
            chunk_overlap=parameters.get('chunk_overlap',200),
            #max_context_length=8000,
            top_k_retrieval=parameters.get('top_k_retrieval',4)
        )

    ### Split between train and test sets
    if run_steps.get('split_train_test', False):
        print("\n--- Splitting between training and testing sets ---")
        parameters = config.get('split_train_test',{})
        split_train_test(
            database_path=parameters.get('database_path'),
            training_set_path=parameters.get('training_set_path'),
            testing_set_path=parameters.get('testing_set_path'),
            train_proportion=parameters.get('train_proportion',0.7),
            random_seed=parameters.get('random_seed',42)
        )   

    ### Build the structured database for the graph RAG
    if run_steps.get('build_structured_database', False):
        print("\n--- Building the structured database for Graph RAG ---")
        parameters = config.get('build_structured_database',{})
        create_structured_database_from_bekanntmachungen(
            data_folder_path=parameters.get('data_folder_path'),
            chunk_size=parameters.get('chunk_size',1000),
            chunk_overlap=parameters.get('chunk_overlap',200),
            output_path=parameters.get('output_path'),
            clean_text=parameters.get('clean_text',True),
            add_tags=parameters.get('add_tags',False)
        )

    ### Build the database of entities and relationships for the graph RAG
    if run_steps.get('build_graph_database', False):
        print("\n--- Building the database of entities and relationships for Graph RAG ---")
        parameters = config.get('build_graph_database',{}) 
        extract_entities_and_relationships(
            database_path=parameters.get('database_path'),
            output_path=parameters.get('output_path'),
            llm_name=parameters.get('llm_name','models/embedding-001'),
            prompt_filepath=parameters.get('prompt_filepath'),
            temperature=parameters.get('temperature',0.1)
        )