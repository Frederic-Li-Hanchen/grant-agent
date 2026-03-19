import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import regex as re
import os
from time import sleep, time
import random  # For random sleep intervals
from utils import load_config_from_yaml
from pdb import set_trace as st
from dotenv import load_dotenv
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel, Field
#import vertexai
from typing import List, Dict, Any, Optional
import json
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from math import floor


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
        text = re.sub(r'(\d)([A-ZÄÖÜa-z])', r'\1 \2', text) # Add space after number if followed by a letter
        text = re.sub(r'(\d+)\s*%', r'\1 %', text) # Ensure space before percentage sign (if not there) for consistency purposes
        text = re.sub(r'([a-z])([A-ZÄÖÜ])', r'\1 \2', text) # Add space between lowercase and uppercase letters
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
        # Sometimes a line break after the title name is missing. Add one to prevent improper title segmentation
        title_pattern = (
            r'(\d(?:\.\d)*\.? (?:Zuwendungszweck(?: Modul)?|Rechtsgrundlage(?:n)?|Zuwendungszweck, Rechtsgrundlage(?:n)?|Gegenstand der Förderung|Zuwendungsempfänger|Zuwendungsvoraussetzungen|'
            r'Art(?:,| und) Umfang(?:,| und) Höhe der (?:Zuwendung|Förderung)|Art und Umfang der Zuwendung|Sonstige Zuwendungsbestimmungen|(?:Antragsv|Förderv|V)erfahren|Projektträger|Einschaltung eines Projektträgers|'
            r'Vorlage und Auswahl von Vorhabenbeschreibungen|Projektskizzen|Förmliche Förderanträge|Kriterien der Begutachtung der Projektskizzen und förmlichen Förderanträge|Verfügbarkeit der Vordrucke|'
            r'Weitere Förderabsichten|Inkrafttreten|Sonstige Nebenbestimmungen|Qualifikationsnachweis|Auswahl- und Entscheidungsverfahren|(?:Einschalten eines Projektträgers und )?Anforderung von Unterlagen|'
            r'Allgemeine Zuwendungsvoraussetzungen(?:/Zuwendungsempfänger(?::)?)?)) ([A-ZÄÖÜß]|\d)'
        )
        text = re.sub(title_pattern, r'\1\n\n\2', text)
        
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
    print(f"Total execution time: {end - start:.2f} seconds.\n")


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
    print(f"Saved {len(training_data_entries)} entries to {output_filepath}\n")


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
    # Note: some documents also have section titles such as A 3, A.4.3, A. 3.2, B.3, B 7.2, B. 5.1, etc.
    #section_pattern = re.compile(r'^\s*(\d(?:\.\d+)*)\.?\s+.*')
    #section_pattern = re.compile(r'^\s*(((?:A|B)[\. ]?\s*)?\d(?:\.\d+)*)\.?\s+.*') 
    section_pattern = re.compile(r'^\s*(((?:A|B)[\. ]?\s*)?[1-9](?:\.[1-9]+)*)\.?\s+.*') 
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

    # If the document has a single section called 'introduction' (may happen for short Bekanntmachungen), change it to 'document'
    if 'introduction' in raw_sections_data and len(raw_sections_data) == 1:
        raw_sections_data['document'] = raw_sections_data.pop('introduction')

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

        # Tag durations with <duration> ... </duration> (e.g. 6 Monate, 2 Jahre, 24-monatiges, etc.)
        # Also includes modifiers such as "bis ... zu", "maximal", "mindestens", "zwischen ... und"
        # Must exclude ages (e.g. "von Kinder unter 3 Jahren", "nicht älter als 65 Jahre") or cycles (e.g. "alle 20 Jahre")
        duration_pattern = r'((?:bis zu\s*|maximal\s*|zwischen \d+ und\s*|(?:von )?\d+ bis\s*|mindestens\s*|höchstens\s*)?(?:\d+|ein|zwei|drei|vier|fünf|vierundzwanzig|sechsunddreizig)\s+(?:Monat(?:e|en)?|Jahr(?:e|en)?|Woche(?:n)?|Tag(?:e|en)?)\b)'
        text = re.sub(duration_pattern, r'<duration>\1</duration>', text)
        duration_pattern2 = r'(\d+-(?:Jahres|jährig(?:\w)*|Monats|monatig(?:\w*)|Tage|tägig(?:\w*)))(-|\s)'
        text = re.sub(duration_pattern2, r'<duration>\1</duration>\2', text)
        duration_pattern3 = r'(?:(?<=Kindern )|(?<=Kinder )|(?<=Kinder unter )|(?<=Kindern unter )|(?<=nicht älter als )|(?<=nicht aelter als )|(?<=alle ))(<duration>)'+duration_pattern+r'(</duration>)'
        text = re.sub(duration_pattern3, r'\2', text)

        # Tag persons with <person> ... </person> (e.g. Dr. Max Mustermann, Prof. Dr. Erika Musterfrau, Frau Erika Musterfrau, etc.)
        # Titles (gender, professor or doctor) are optional, but at least one must be present to avoid false positives
        # Name detection is achieved by a regular expression of type (AB?C?|A?BC?|AB?C) where A, B, C are the title patterns
        # NOTE: in some rare cases, the person name is not having any title preceding it and is therefore missed. To avoid false positives, these are ignored.
        title_pattern = r'(?:(?:Herr|Frau|Herrn|Mr\.?|Ms\.?)\s*)'
        professor_pattern = r'(?:Prof\.\s*|Professor(?:in)?\s*)'
        doctor_pattern = r'(?:Dr\.\s*(?:-Ing\.|-ing\.|rer\.\s*nat\.)?\s*)'
        firstname_pattern = r'(?:[A-ZÄÖÜ][a-zäöüß]+)?(?:(?:-[a-zA-ZÄÖÜ]|\s[A-ZÄÖÜ])[a-zäöüß]+)?\s*'
        lastname_pattern = r'(?:[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?|[A-ZÄÖÜ]+(?:(?:-|s)[A-ZÄÖÜ]+)?)'
        person_pattern = r'(\b(?:' + title_pattern + professor_pattern + r'?' + doctor_pattern + r'?|' + title_pattern + r'?' + professor_pattern + doctor_pattern + r'?|' + title_pattern + r'?' + professor_pattern + r'?'+ doctor_pattern + r')' + firstname_pattern + r'?' + lastname_pattern + r'\b)'
        text = re.sub(person_pattern, r'<person>\1</person>', text)
        # The person tagging may have caught some extra words that are not names (e.g. "Telefon", "E-Mail", etc.), so we remove those from within the tags
        exclusion_pattern = r'(?:\s*)(Telefon|Beratungstelefon|Telefonnummer|E-Mail|Email|Mail|Fax|Telefax|Adresse|Anschrift|Straße|Str|Platz|Weg|Hausnummer|Nr|Postleitzahl|Ort|Stadt|Bundesland|Land|Webseite|Website|Internet|Department|Abteilung)'
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

    print(f'\nCreating the structured database from {data_folder_path} with chunk size {chunk_size} and overlap {chunk_overlap} for Graph RAG ...\n')
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
        if "bekanntmachung" in file_name.lower() and not "aenderung" in file_name.lower() and not "%C3%84nderung" in file_name and not "vergabe" in file_name.lower() and not 'aufhebung' in file_name.lower() and not "berichtigung" in file_name.lower():
            document_type = "bekanntmachung"
        elif "aenderung" in file_name.lower() or "%C3%84nderung" in file_name or "berichtigung" in file_name.lower():
            document_type = "modification"
        elif "vergabe" in file_name.lower():
            document_type = "award"
        elif "aufhebung" in file_name.lower():
            document_type = "cancellation"
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
                            chunk.metadata["id"] = unique_id.replace(' ','')    
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

### Function that assigns a metadata tag 'topics' to all chunks of a structured database for efficient grouping when calling a LLM for entity extraction
# Input:
# - [str] database_path: the path to the jsonl file containing the structured database of chunks
# - [str] output_path: the path where to save the output database of structured chunks with augmented metadata
# Output:
# - None
def assign_topic_to_chunk(database_path: str, output_path: str):

    start_time = time()

    # Load the jsonl file
    with open(database_path, 'r', encoding='utf-8') as f:
        database = f.readlines()
    
    # Define the heuristics to determine whether a chunk should be tagged with the corresponding topic
    heuristics = {}
    # Objective heuristics
    # This filtering is done by using section indices: the objective is always described either in 'document', 'introduction', or any section under '1' or '2'.
    # Any chunk from a section whose title contains "Zuwendungszweck" is kept. Otherwise, a keyword-based filtering is added.
    objective = r'(?:Bereich|Ziel | ziehlt)'
    heuristics['objective'] = {
        'sections': [r'document', r'introduction', r'^1(?:\.\d)*\s(?=Zuwendungszweck)', r'^1(?:\.\d)*\s(?!Zuwendungszweck)', r'^2(?:\.\d)*'], 
        'added_filters': {
            're': {
                r'document': [objective], 
                r'introduction': [objective], 
                r'^1(?:\.\d)*\s(?=Zuwendungszweck)': [], # Any chunk from section Zuwendungszweck is kept
                r'^1(?:\.\d)*\s(?!Zuwendungszweck)': [objective],  
                r'^2(?:\.\d)*': [objective]
            },
            'operator': 'or'
        }
    }

    # Inclusion criteria heuristics
    # Inclusion criteria are mostly found in Section 3, with supporting information sometimes found in sections 4, 2, and 1
    # Chunks from sections 4, 2 and 1 are filtered by keywords, while any chunk from section 3 is kept
    inclusion = r'(?:L(?:a|ä)nd(?:er|ern)? |qualifiziert|(?:antrags)?berechtigt|Einschlusskriterien|Teilnahmebedingungen)'
    heuristics['inclusion_criteria'] = {
        'sections': [r'^1(?:\.\d)*', r'^2(?:\.\d)*', r'^3(?:\.\d)*', r'^4(?:\.\d)*'],
        'added_filters': {
            're': {
                r'^1(?:\.\d)*': [inclusion],
                r'^2(?:\.\d)*': [inclusion],
                r'^3(?:\.\d)*': [], # Any chunk from Section 3.* is kept
                r'^4(?:\.\d)*': [inclusion]
            },
            'operator': 'or'
        }
    }

    # Exclusion criteria heuristics
    # Exclusion criteria are mostly found in sections 2, 3, 4 and 5.
    # A keyword filtering is added for the chunks from these sections.
    exclusion = r'(?:ausschließlich |Ausschlusskriterien|nicht älter als|nicht finanziert|nicht (\w)*berechtigt)'
    heuristics['exclusion_criteria'] = {
        'sections': [r'^2(?:\.\d)*', r'^3(?:\.\d)*', r'^4(?:\.\d)*', r'^5(?:\.\d)*'],
        'added_filters': {
            're': {
                r'^2(?:\.\d)*': [exclusion],
                r'^3(?:\.\d)*': [exclusion],
                r'^4(?:\.\d)*': [exclusion],
                r'^5(?:\.\d)*': [exclusion]
            },
            'operator': 'or'
        }
    }

    # Deadline heuristics: based on the presence of both <date> tags and specific keywords.
    deadline1 = r'<date>.*</date>'
    deadline2 = r'(?:spätestens|vor dem|bis zum|deadline|Frist |Stichtag )'
    heuristics['deadline'] = {'re': [deadline1, deadline2], 'operator': 'and'}

    # Max funding heuristics: based on the presence of both <amount> tags and specific keywords.
    max_funding1 = r'<amount>.*</amount>'
    max_funding2 = r'(?:bis zu|maximal|höchstens|Förderhöchstbetrag |pro Vorhaben|pro (?:Industriep|Forschungsp|Verbundp|Entwicklungsp|P)rojekt |Fördervolumen |pro (?:Projektv|V)erbund|pro (?:Projektp|P)artner)'
    heuristics['max_funding'] = {'re': [max_funding1, max_funding2], 'operator': 'and'}

    # Max duration heuristics: based on the presence of both <duration> tags and specific keywords.
    max_duration1 = r'<duration>.*</duration>'
    max_duration2 = r'(?:Laufzeit|(?:Industriep|Forschungsp|Verbundp|Entwicklungsp|P)rojekt|Dauer|gefördert)'
    heuristics['max_duration'] = {'re': [max_duration1, max_duration2], 'operator': 'and'}

    # Procedure heuristics
    # Any chunk from a section containing "Verfahren" or variants in its title are kept.
    # Otherwise, chunks from other sections must contains the keywords einstufig|zweistufig|dreistufig|vierstufig.
    procedure = r'(?:(?:E|e)instufig|(?:Z|z)weistufig|(?:D|d)reistufig|(?:V|v)ierstufig)|1-stufig|2-stufig|3-stufig|4-stufig|Verfahrensstufe'
    heuristics['procedure'] = {
        'sections': [r'(?<!(?:1|2)(?:\.\d)*.*)\s(?:V|v|\w*v)erfahren',r'.*'],
        'added_filters': {
            're': {
                r'(?<!(?:1|2)(?:\.\d)*.*)\s(?:V|v|\w*v)erfahren': [],
                r'.*': [procedure]
            },
            'operator': 'or'
        }
    }

    # Contact heuristics
    # Chunks that contain specific combinations of <person>, <phone> and <email> tags, with or without specific keywords.
    contact1 = r'(?s)^(?=.*<person>)(?=.*<phone>).*'
    contact2 = r'(?s)^(?=.*<person>)(?=.*<email>).*'
    contact3 = r'(?s)^(?=.*<phone>)(?=.*<email>).*'
    contact4 = r'(?s)^(?=.*(weitere Informationen|Kontakt|kontaktieren|Ansprechpartner))(?=.*<email>).*'
    contact5 = r'(?s)^(?=.*(weitere Informationen|Kontakt|kontaktieren|Ansprechpartner))(?=.*<phone>).*'
    heuristics['contact'] = {'re': [contact1, contact2, contact3, contact4, contact5], 'operator': 'or'}

    # Misc heuristics
    # Includes rules to catch chunks discussing Förderquote, de-minimis, preferences regarding consortium participants, geographical requirements regarding result exploitation.
    misc1 = r'(?s)^(?=.*(Förderquote|Verbundförderquote|förderfähige Kosten|Wissenschaftseinrichtungen|Fraunhofer|Hochschulen|SME))(?=.*<percentage>).*' # Förderquote
    misc2 = r'(?:d|D)e(?:-|\s)(?:M|m)inimis'
    misc3 = r'(?s)^(?=.*besonders)(?=.*aufgefordert)(?=.*beteiligen).*'
    misc4 = r'(?s)^(?=.*Ergebnisse)(?=.*nur in ).*'
    heuristics['misc'] = {'re': [misc1, misc2, misc3, misc4], 'operator': 'or'}

    # Loop on the chunks from the database
    topics = heuristics.keys()
    chunk_idx = 0
    output_file = open(output_path, 'a', encoding='utf-8')

    for list_item in database:
        # Print progress
        chunk_idx += 1
        if chunk_idx%1000 == 0:
            print(f"Processed the {chunk_idx}th database entry ...")
        # Load the chunk
        chunk = json.loads(list_item)
        # Get text and meta-data
        text = chunk.get('text','')
        chunk_id = chunk.get('metadata',{}).get('id','not found')
        section_title = chunk.get('metadata',{}).get('section_title','not found')
        # Set the chunk metadata "topics" field to empty list
        if chunk.get('metadata',{}):
            chunk['metadata']['topics'] = []
        else:
            print(f'    Metadata missing for chunk number {chunk_id}. Skipping ...')
            continue

        # Loop on the topics
        for topic in topics:
            rules = heuristics[topic].get('re',[])
            sections = heuristics[topic].get('sections',[])
            operator = heuristics[topic].get('operator','')

            if rules: # Regular expressions to look for relevant chunks
                conditions = [isinstance(re.search(e,text),re.Match) for e in rules]
                if (operator == 'and' and sum(conditions) == len(conditions)) or (operator == 'or' and sum(conditions)>=1): # Either all are at least one condition must be met depending on the operator
                    # Add the topic tag to the chunk metadata
                    chunk['metadata']['topics'] += [topic]
                
            elif sections: # Section titles to look for relevant chunks
                # Get additional regular expression rules
                added_filters = heuristics[topic].get('added_filters', {})
                # Keep only the (sub)title with the deepest level if not an annex (Anlage)
                section_titles = section_title.split(';')
                if section_titles[0] != 'Anlage':
                    section_title = section_title.split(';')[-1].strip()
                else:
                    section_title = 'Anlage'
                # Check if the chunk belongs to any of the allowed sections
                conditions = [isinstance(re.search(e,section_title), re.Match) for e in sections]
                if sum(conditions)>=1:
                    # If added filters are present, check if the condition(s) for the corresponding section is (are) fulfiled
                    if added_filters:
                        # Get the correct set of conditions by obtaining the (first) element of sections that returned true
                        idx = conditions.index(True)
                        additional_rules = added_filters.get('re', []).get(sections[idx], [])
                        # Check that all additional conditions are fulfilled
                        if additional_rules:
                            additional_conditions = [isinstance(re.search(e,text),re.Match) for e in additional_rules]
                            if added_filters.get('operator','') == 'and':
                                keep_chunk = (sum(additional_conditions)==len(additional_conditions))
                            elif added_filters.get('operator','') == 'or':
                                keep_chunk = (sum(additional_conditions)>=1)
                        else: # No additional rule for this specific section, the chunk is always kept
                            keep_chunk = True
                    else: # No added filter present
                        keep_chunk = True
                    if keep_chunk:
                        # Add the topic tag to the chunk metadata
                        chunk['metadata']['topics'] += [topic]

        # Add the updated chunk to the new database
        json_line = json.dumps(chunk, ensure_ascii=False)
        output_file.write(json_line+'\n')

    # Close the output file
    output_file.close()
    end_time = time()
    print(f'Topic tagging process completed in {end_time-start_time:.2f} seconds.')


### Function that extracts the entities and relationships from the structured database.
### Entities and relationships are represented as triplets of (subject, predicate, object).
### Extracts structural triplets (document, HAS_SECTION, section) by parsing the database file.
### Extracts conceptual triplets (e.g. (document, DEFINES_OBJECTIVE, objective)) by a LLM call.
### Save all extracted concepts into a jsonl file with fields ["subject_type", "subject_value", "predicate", "object_type", "object_value", "source_document_id", "source_section_title"].
# Input:
# - [str] database_path: path to the jsonl file containing the structured database of chunks
# - [str] output_path: path to the jsonl file where the database of entities and relationships should be saved.
# - [str] prompt_filepath: path to the file containing the LLM prompt for concept extraction.
# - [str] llm_name: name of the LLM to be used for triplet extraction.
# - [int] chunk_batch: number of chunks with at least one associated topic to be provided at once to the LLM.
# - [float] temperature: temperature parameter of the LLM to be used for entity extraction
# Output:
# - None
def extract_entities_and_relationships(
        database_path: str,
        output_path: str, 
        prompt_filepath: str, 
        llm_name: str='gemini-2.5-flash',
        chunk_batch: int=4,
        temperature: float=0.1):
    
    start_time = time()
    print('\nExtracting entities and relationships from the structured database ...')

    # Define the topics from which related information should be extracted
    topics = {
        "objective": "the main topics of the projects to be financed by the call.",
        "inclusion_criteria": "any criteria that applicants for the call must fulfill.",
        "exclusion_criteria": "any criteria that prevents applicants from applying to the call.",
        "deadline": "the deadline before which the applications must be submitted.",
        "max_funding": "the maximum amount of funding allowed, either per applying participant or consortium.", 
        "max_duration": "the maximum duration allowed for the project.",
        "procedure": "the procedure to follow for the submission process.",
        "contact": "the person(s) or institutions to contact to obtain more information about the call.",
        "misc": "additional miscelleanous information such as whether de-minimis applies, specified 'Förderquote' (funding rate), geographical restrictions on the exploitation of results."
    }

    # Load the jsonl file containing the chunks
    with open(database_path, 'r', encoding='utf-8') as f:
        chunks_database = f.readlines()

    # Get the list of all Bekanntmachungen file names from the loaded json file
    file_list = [json.loads(e).get('metadata', {}).get('document_id', None) for e in chunks_database]
    file_list = list(set(file_list))

    # --- Intelligent Resuming Logic ---
    # Determine all the filenames that are existing in the database
    if os.path.exists(output_path):
        print(f"Output file found at {output_path}. Loading existing triplets to prevent duplicates.")
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_triplets = f.readlines()
    else:
        existing_triplets = []

    existing_filenames = []
    for triplet in existing_triplets:
        current_chunk_id = json.loads(triplet).get('chunk_id', None)
        # Extract filename from chunk_id
        pos = current_chunk_id.find('_sec')
        current_filename = current_chunk_id[:pos]+'.txt'
        if current_filename and current_filename not in existing_filenames:
            existing_filenames.append(current_filename)

    # --- Determine the files to be processed ---
    files_to_process = [e for e in file_list if e not in existing_filenames]
    files_to_process.sort()
    print(f'Found {len(existing_filenames)} existing files and {len(files_to_process)} files to process.')

    # --- Initialise the LLM for conceptual chunks extraction ---
    load_dotenv()
    graph_rag_api_key = os.getenv("GRAPH_RAG_GEMINI_API_KEY")
    if graph_rag_api_key is None:
        raise ValueError("GRAPH_RAG_GEMINI_API_KEY not found among the environment variables defined in .env")
    llm = ChatGoogleGenerativeAI(model=llm_name, temperature=temperature, google_api_key=graph_rag_api_key)

    # --- Load the prompt template from the provided json file ---
    with open(prompt_filepath,'r',encoding='utf-8') as f:
        prompt_template_string = json.load(f)['prompt_template']
    prompt_template = ChatPromptTemplate.from_template(prompt_template_string)

    # --- Define the Pydantic model for the expected output ---
    class Triplet(BaseModel):
        """A structured representation of a relationship triplet."""
        subject_type: str = Field(description="The type of the subject that takes values among the previously defined entities.")
        subject_value: str = Field(description="The value taken by the subject, of maximum 3 sentences or 150 words.")
        predicate: str = Field(description="The predicate or type of the relationship.")
        object_type: str = Field(description="The type of the object that takes values among the previously defined entities.")
        object_value: str = Field(description="The value taken by the object, of maximum 3 sentences or 150 words.")
        chunk_id: str = Field(description="The ID of the chunk the triplet is extracted from.")
    # Define a class that is a triplet list:
    class TripletList(BaseModel):
        triplets: List[Triplet] = Field(description="A list of triplets related to one of the following fields: 'objective', 'inclusion_criteria', 'exclusion_criteria', 'deadline', 'max_funding', 'max_duration', 'procedure', 'contact', 'misc'.")
    # Define the chain to parse a list of triplets
    parser = PydanticOutputParser(pydantic_object=TripletList)    
    # Create a chain with a fallback for parsing errors.
    # If the initial parsing fails, it passes the output and the error to a fixing chain.
    fixing_prompt_template = ChatPromptTemplate.from_template(
        "Fix the following output to conform to the format instructions. Do not add any other text.\n\n"
        "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        "FAILED OUTPUT:\n{completion}\n\n"
        "ERROR:\n{error}"
    )
    fixing_chain = fixing_prompt_template | llm | parser
    chain = (prompt_template | llm | parser).with_fallbacks(fallbacks=[fixing_chain], exception_key="error", input_key="completion")
    
    # --- Iterate over the documents to process ---
    for file_idx, file_name in enumerate(files_to_process):
        file_start_time = time()
        # --- Retrieve the chunks that are associated with the file ---
        chunks_for_current_file = [json.loads(e) for e in chunks_database if json.loads(e).get('metadata', {}).get('document_id', None) == file_name]
        # --- Extract the document structure triplets ---
        document_structure_triplets = []
        already_seen = []
        for chunk in chunks_for_current_file:
            # Retrieve the title of the section at the deepest level only
            titles = chunk["metadata"]["section_title"].split("; ")
            deepest_title = titles[-1] if titles else ""
            if not deepest_title in already_seen:
                already_seen.append(deepest_title)
                new_triplet = {
                    "subject_type": "DOCUMENT",
                    "subject_value": chunk["metadata"]["document_id"],
                    "predicate": "HAS_SECTION",
                    "object_type": "SECTION",
                    "object_value": deepest_title,
                    "chunk_id": chunk["metadata"]["id"]
                }
                document_structure_triplets.append(new_triplet)
        # --- Extract the conceptual chunks ---
        conceptual_triplets = []
        # Among the chunks of the document, keep only the ones that have a non-empty "topics" in "metadata"
        chunks_for_current_file_with_topics = [e for e in chunks_for_current_file if e.get('metadata', {}).get('topics', None)]
        # Loop on the chunks by agregating them by batches of chunk_batch
        if chunks_for_current_file_with_topics:
            idx = 0
            while idx < len(chunks_for_current_file_with_topics):
                end_idx = min(idx+chunk_batch, len(chunks_for_current_file_with_topics))
                current_chunks = chunks_for_current_file_with_topics[idx:end_idx]
                # Retrieve all information required to be input to the prompt
                document_title = current_chunks[0]["metadata"]["document_id"] # The document title is the same for all chunks
                current_topics = [] # The different topics that are contained in the chunks
                current_section_titles = [] # The different section titles
                current_chunk_id = [] # The different chunk IDs
                for chunk in current_chunks:
                    titles = chunk["metadata"]["section_title"].split("; ")
                    deepest_title = titles[-1] if titles else ""
                    current_topics += chunk["metadata"]["topics"]
                    current_section_titles.append(deepest_title)
                    current_chunk_id.append(chunk["metadata"]["id"])
                # Remove duplicated topics
                current_topics = list(set(current_topics))
                # Build the list of topics in a single string with proper list formatting
                formatted_topic_list = "\n".join([f"- **{topic}**: {topics[topic]}" for topic in current_topics])
                # Building the text of the chunks in a single string with proper markdown formatting
                formatted_chunk_text = "\n"
                for chunk_nb, chunk in enumerate(current_chunks):
                    formatted_chunk_text += f"\n**Chunk {chunk_nb}**\n*Chunk ID:* {current_chunk_id[chunk_nb]}\n*Section title:* {current_section_titles[chunk_nb]}\n*Text:* {chunk['text']}\n\n---\n"
                formatted_chunk_text += "\n"
                # Create the prompt input
                prompt_input = {
                    "nb_chunks": len(current_chunks),
                    "topic_list": formatted_topic_list,
                    "max_nb_chunks": floor(2.5*len(current_topics)),
                    "document_title": document_title,
                    "chunk_data": formatted_chunk_text,
                    "format_instructions": parser.get_format_instructions(),
                }
                # Call the LLM with the defined prompt
                try:
                    # The parser returns a Pydantic object. It might be None if the LLM returns nothing.
                    result_triplet_list = chain.invoke(prompt_input)
                    
                    if result_triplet_list and result_triplet_list.triplets:
                        for triplet in result_triplet_list.triplets:
                            new_triplet = triplet.model_dump() # Convert Pydantic model to dictionary
                            # Save the new triplet in the output jsonl file
                            with open(output_path, 'a', encoding='utf-8') as f:
                                json_line = json.dumps(new_triplet, ensure_ascii=False)
                                f.write(json_line + '\n')
                    else:
                        print(f"  - No conceptual triplet found for chunks {idx+1} to {idx+chunk_batch} of document {document_title}.")
                except ResourceExhausted as e:
                    print(f"  - Resource exhausted for chunks {idx+1} to {idx+chunk_batch} of document {document_title}. Waiting 60 seconds before retrying. Error: {e}")
                    sleep(60)
                    # You might want to re-add the chunk to a queue to retry later
                except Exception as e:
                    print(f"  - Error generating conceptual triplet for chunks {idx+1} to {idx+1+chunk_batch} of document {document_title}: {e}. Skipping.")
                # Increment the batch index
                idx += chunk_batch
            # --- Save all triplets extracted for the current document ---
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for triplet in document_structure_triplets:
                    json_line = json.dumps(triplet, ensure_ascii=False)
                    output_file.write(json_line + '\n')
                for triplet in conceptual_triplets:
                    json_line = json.dumps(triplet, ensure_ascii=False)
                    output_file.write(json_line + '\n')
        file_end_time = time()
        print(f'Processed file {file_idx+1}/{len(files_to_process)} ({file_name}) in {file_end_time-file_start_time:.2f} seconds')
    end_time = time()
    print(f"Entities and relationships extracted and saved in {output_path} in {end_time - start_time:.2f} seconds.\n")


# =============================================
# === Pre-processing of the Graph Database ===
# =============================================

### Function that expands conjunction triplets into individual triplets.
### Values of the targeted entity types that list multiple entities joined by a conjunction marker
### (e.g. "Agency A und Agency B") are split into their constituents, and one triplet is produced
### per constituent combination so that each node in the graph represents a single entity.
### Only values where every part produced by the split contains at least min_part_words words are
### split, to avoid incorrectly splitting institution names that legitimately contain "und"
### (e.g. "Bundesministerium für Bildung und Forschung", where "Forschung" is a single word).
# Input:
# - [str] input_path: path to the jsonl file containing the raw graph database of triplets
# - [str] output_path: path to the jsonl file where the expanded graph database should be saved
# - [List[str]] entity_types_to_split: entity types for which conjunction values should be split
# - [List[str]] conjunction_markers: substrings used to detect and split conjunction values
# - [int] min_part_words: minimum number of words each split part must have for the split to apply
# Output:
# - None
def split_conjunction_triplets(
        input_path: str,
        output_path: str,
        entity_types_to_split: List[str] = None,
        conjunction_markers: List[str] = None,
        min_part_words: int = 2
) -> None:

    if entity_types_to_split is None:
        entity_types_to_split = ['FUNDING_BODY', 'APPLICANT', 'PERSON', 'LOCATION']
    if conjunction_markers is None:
        conjunction_markers = [' und ', ' and ', ' sowie ']

    start_time = time()
    print('\nSplitting conjunction triplets in the graph database ...')

    with open(input_path, 'r', encoding='utf-8') as f:
        triplets = [json.loads(line) for line in f]
    print(f'Loaded {len(triplets)} triplets from {input_path}.')

    # Split a value on all conjunction markers and return the list of constituent parts.
    # Returns the original value as a single-element list if the split does not apply.
    def split_value(value: str) -> List[str]:
        result = value
        for marker in conjunction_markers:
            result = result.replace(marker, '\x00')  # null byte as internal separator
        parts = [p.strip() for p in result.split('\x00') if p.strip()]
        return parts if len(parts) >= 2 else [value]

    # A value is a splittable conjunction if every constituent part has at least min_part_words words
    # and no part ends with a hyphen. A trailing hyphen indicates a German compound abbreviation
    # (e.g. "Luft- und Raumfahrt" = "Luftfahrt und Raumfahrt"), not a conjunction of two entities.
    def is_splittable(entity_type: str, value: str) -> bool:
        if entity_type not in entity_types_to_split:
            return False
        parts = split_value(value)
        return (len(parts) >= 2
                and not any(p.endswith('-') for p in parts)
                and all(len(p.split()) >= min_part_words for p in parts))

    output_triplets = []
    n_original_split = 0

    for triplet in triplets:
        split_subj = is_splittable(triplet['subject_type'], triplet['subject_value'])
        split_obj = is_splittable(triplet['object_type'], triplet['object_value'])

        if not split_subj and not split_obj:
            output_triplets.append(triplet)
            continue

        n_original_split += 1
        subj_parts = split_value(triplet['subject_value']) if split_subj else [triplet['subject_value']]
        obj_parts = split_value(triplet['object_value']) if split_obj else [triplet['object_value']]

        for s in subj_parts:
            for o in obj_parts:
                new_triplet = dict(triplet)
                new_triplet['subject_value'] = s
                new_triplet['object_value'] = o
                output_triplets.append(new_triplet)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for triplet in output_triplets:
            f_out.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    end_time = time()
    print(f'Split {n_original_split} conjunction triplet(s): {len(triplets)} → {len(output_triplets)} total triplets.')
    print(f'Split graph database saved to {output_path} in {end_time - start_time:.2f} seconds.\n')


# =============================================
# === Canonicalisation of the Graph Database ===
# =============================================

### Function that canonicalises entity values in the graph database by clustering semantically similar
### surface forms within each entity type and replacing them with a single canonical representative.
### Only entity types listed in entity_types_to_canonicalise or entity_types_normalise_only are
### processed; structural types such as DOCUMENT and SECTION are left unchanged.
### Two processing strategies are applied depending on the entity type:
###   - Normalise-only (entity_types_normalise_only, default: PERSON): strips suffixes via
###     department_separators (e.g. "Person A (technical contact)" -> "Person A") but performs no
###     clustering. This prevents false-positive merges between different people, where embedding
###     and token-overlap similarity are unreliable for short proper nouns.
###   - Full pipeline (entity_types_to_canonicalise, default: FUNDING_BODY, APPLICANT, LOCATION):
###     1. Conjunction filtering: values listing multiple entities are kept verbatim.
###     2. Department normalisation: suffixes are stripped before embedding.
###     3. Combined-distance clustering: merges when cosine similarity >= similarity_threshold OR
###        Jaccard token similarity >= token_overlap_threshold.
### The canonical name for each cluster is the most frequent normalised surface form.
### A mapping file is saved alongside the output for inspection.
# Input:
# - [str] input_path: path to the jsonl file containing the raw graph database of triplets
# - [str] output_path: path to the jsonl file where the canonicalised graph database should be saved
# - [str] mapping_output_path: path to the json file where the canonicalisation mapping should be saved
# - [str] embedding_model_path: sentence-transformer model used to embed entity surface forms
# - [float] similarity_threshold: cosine similarity above which two surface forms are merged (e.g. 0.75)
# - [float] token_overlap_threshold: Jaccard token similarity above which two forms are merged (e.g. 0.5)
# - [List[str]] department_separators: substrings that separate a name from a department or role suffix
# - [List[str]] conjunction_markers: substrings indicating a value lists multiple entities
# - [List[str]] entity_types_normalise_only: entity types that get suffix stripping but no clustering
# - [List[str]] entity_types_to_canonicalise: entity types that get the full clustering pipeline
# Output:
# - None
def canonicalise_graph_database(
        input_path: str,
        output_path: str,
        mapping_output_path: str,
        embedding_model_path: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        similarity_threshold: float = 0.75,
        token_overlap_threshold: float = 0.5,
        department_separators: List[str] = None,
        conjunction_markers: List[str] = None,
        entity_types_normalise_only: List[str] = None,
        entity_types_to_canonicalise: List[str] = None
) -> None:

    import numpy as np
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    if entity_types_normalise_only is None:
        entity_types_normalise_only = ['PERSON']
    if entity_types_to_canonicalise is None:
        entity_types_to_canonicalise = ['FUNDING_BODY', 'APPLICANT', 'LOCATION']
    if department_separators is None:
        department_separators = [' – ', ' - ', ' (']
    if conjunction_markers is None:
        conjunction_markers = [' und ', ' and ', ' sowie ']

    start_time = time()
    print('\nCanonicalising entity values in the graph database ...')

    # --- Load all triplets ---
    with open(input_path, 'r', encoding='utf-8') as f:
        triplets = [json.loads(line) for line in f]
    print(f'Loaded {len(triplets)} triplets from {input_path}.')

    # --- Count occurrences of each (type, value) pair across subjects and objects ---
    # Used to elect the most frequent normalised form as the canonical name per cluster.
    value_counts: Dict[tuple, int] = {}
    for triplet in triplets:
        for side_type, side_value in [
            (triplet['subject_type'], triplet['subject_value']),
            (triplet['object_type'], triplet['object_value'])
        ]:
            key = (side_type, side_value)
            value_counts[key] = value_counts.get(key, 0) + 1

    # --- Helper: strip department-level suffix ---
    # Keeps only the part of the value before the first department separator.
    # E.g. "DLR – Projektträger Energie" -> "DLR"
    def normalise_value(value: str) -> str:
        for sep in department_separators:
            idx = value.find(sep)
            if idx > 0:
                return value[:idx].strip()
        return value

    # --- Helper: detect conjunction values ---
    # A value is treated as a conjunction if splitting on a marker yields >=2 parts each
    # containing at least 2 words. The 2-word minimum prevents false positives on legitimate
    # institution names that contain "und" (e.g. "Bundesministerium für Bildung und Forschung",
    # where "Forschung" is a single word after the split).
    def is_conjunction(value: str) -> bool:
        for marker in conjunction_markers:
            parts = value.split(marker)
            if len(parts) >= 2 and all(len(p.strip().split()) >= 2 for p in parts):
                return True
        return False

    # --- Helper: token-level Jaccard similarity ---
    def jaccard_similarity(a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a and not tokens_b:
            return 1.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    # --- Load the embedding model ---
    print(f'Loading embedding model from {embedding_model_path} ...')
    model = SentenceTransformer(embedding_model_path)

    # --- Build the canonical mapping ---
    # canonical_map[(entity_type, original_surface_form)] -> canonical_value
    canonical_map: Dict[tuple, str] = {}

    # --- Normalise-only pass (no clustering) ---
    # For short proper-noun types like PERSON, embedding and token-overlap similarity are unreliable
    # (e.g. "Dr. A Müller" and "Dr. B Müller" score high on both metrics despite being different
    # people). Only suffix stripping is applied: "Person A (technical contact)" -> "Person A".
    for entity_type in entity_types_normalise_only:
        originals = [v for (t, v) in value_counts if t == entity_type]
        if not originals:
            print(f'  - {entity_type}: no values found, skipping.')
            continue
        n_changed = 0
        for v in originals:
            canonical = normalise_value(v)
            canonical_map[(entity_type, v)] = canonical
            if canonical != v:
                n_changed += 1
        print(f'  - {entity_type}: {len(originals)} values normalised, {n_changed} suffix(es) stripped (no clustering).')

    # --- Full clustering pass ---
    for entity_type in entity_types_to_canonicalise:
        originals = [v for (t, v) in value_counts if t == entity_type]

        if not originals:
            print(f'  - {entity_type}: no values found, skipping.')
            continue

        # Step 1 — Conjunction filtering: keep conjunction values verbatim
        conjunction_vals = [v for v in originals if is_conjunction(v)]
        cluster_vals = [v for v in originals if not is_conjunction(v)]
        for v in conjunction_vals:
            canonical_map[(entity_type, v)] = v
        if conjunction_vals:
            print(f'  - {entity_type}: {len(conjunction_vals)} conjunction value(s) kept as-is.')

        if not cluster_vals:
            continue

        # Step 2 — Department normalisation: map each original to its institution-level form
        # Multiple originals may collapse to the same normalised form here (free merge).
        normalised = {v: normalise_value(v) for v in cluster_vals}

        # Aggregate occurrence counts at the normalised-form level
        norm_counts: Dict[str, int] = {}
        for v in cluster_vals:
            n = normalised[v]
            norm_counts[n] = norm_counts.get(n, 0) + value_counts.get((entity_type, v), 0)
        unique_norms = list(norm_counts.keys())

        if len(unique_norms) == 1:
            canonical = unique_norms[0]
            for v in cluster_vals:
                canonical_map[(entity_type, v)] = canonical
            print(f'  - {entity_type}: all {len(cluster_vals)} values normalise to a single entity.')
            continue

        print(f'  - {entity_type}: {len(cluster_vals)} values → {len(unique_norms)} normalised forms — embedding and clustering ...')

        # Step 3 — Combined-distance clustering on normalised forms
        # Embed normalised values (L2-normalised → dot product = cosine similarity)
        embeddings = model.encode(unique_norms, show_progress_bar=False, normalize_embeddings=True)
        cosine_sim = np.dot(embeddings, embeddings.T)
        cosine_dist = np.clip(1.0 - cosine_sim, 0, 2)
        np.fill_diagonal(cosine_dist, 0.0)

        # Normalise each distance by its respective merge threshold so that the clustering
        # cut-point is always 1.0. combined_dist[i,j] <= 1.0 iff cosine OR Jaccard criterion met.
        emb_dist_thresh = 1.0 - similarity_threshold
        tok_dist_thresh = 1.0 - token_overlap_threshold
        cosine_dist_norm = cosine_dist / emb_dist_thresh

        n_vals = len(unique_norms)
        jaccard_dist_norm = np.ones((n_vals, n_vals), dtype=float)
        for i in range(n_vals):
            for j in range(i + 1, n_vals):
                j_sim = jaccard_similarity(unique_norms[i], unique_norms[j])
                jaccard_dist_norm[i, j] = jaccard_dist_norm[j, i] = (1.0 - j_sim) / tok_dist_thresh
        np.fill_diagonal(jaccard_dist_norm, 0.0)

        combined_dist = np.minimum(cosine_dist_norm, jaccard_dist_norm)
        np.fill_diagonal(combined_dist, 0.0)
        condensed = squareform(combined_dist, checks=False)

        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=1.0, criterion='distance')

        # For each cluster, elect the normalised form with the highest aggregated count
        clusters: Dict[int, List[str]] = {}
        for norm_val, label in zip(unique_norms, labels):
            clusters.setdefault(int(label), []).append(norm_val)

        print(f'    -> {len(unique_norms)} normalised forms merged into {len(clusters)} canonical entities.')

        for cluster_norms in clusters.values():
            canonical = max(cluster_norms, key=lambda nv: norm_counts.get(nv, 0))
            for norm_val in cluster_norms:
                for v in cluster_vals:
                    if normalised[v] == norm_val:
                        canonical_map[(entity_type, v)] = canonical

    n_types = len(entity_types_normalise_only) + len(entity_types_to_canonicalise)
    print(f'Canonical mapping built: {len(canonical_map)} entries across {n_types} entity types.')

    # --- Save the mapping for inspection ---
    mapping_serialisable = {f"{t}||{v}": c for (t, v), c in canonical_map.items()}
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_serialisable, f, ensure_ascii=False, indent=2)
    print(f'Mapping saved to {mapping_output_path}.')

    # --- Apply mapping and write canonicalised output ---
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for triplet in triplets:
            subj_key = (triplet['subject_type'], triplet['subject_value'])
            obj_key = (triplet['object_type'], triplet['object_value'])
            triplet['subject_value'] = canonical_map.get(subj_key, triplet['subject_value'])
            triplet['object_value'] = canonical_map.get(obj_key, triplet['object_value'])
            f_out.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    end_time = time()
    print(f'Canonicalised graph database saved to {output_path} in {end_time - start_time:.2f} seconds.\n')


# ==========================================
# === Filtering of the Graph Database ===
# ==========================================

### Function that removes triplets whose subject or object value is structurally invalid for its
### declared entity type. Filtering is applied only to the entity types listed in entity_types_to_filter.
### The following rules flag a value as invalid:
###   - Matches a date pattern (German DD.MM.YYYY or ISO YYYY-MM-DD)
###   - Matches a URL or e-mail pattern
###   - Consists entirely of digits, punctuation, or numeric symbols
###   - Exceeds max_words words (indicates a descriptive sentence rather than an entity name)
### Rejected triplets are written to a separate file for manual inspection.
# Input:
# - [str] input_path: path to the jsonl file containing the graph database to filter
# - [str] output_path: path to the jsonl file where the filtered graph database should be saved
# - [str] rejected_output_path: path to the jsonl file where rejected triplets should be saved
# - [List[str]] entity_types_to_filter: entity types to apply the validation rules to
# - [int] max_words: maximum number of words allowed in a valid entity value
# Output:
# - None
def filter_graph_database(
        input_path: str,
        output_path: str,
        rejected_output_path: str,
        entity_types_to_filter: List[str] = None,
        max_words: int = 20
) -> None:

    if entity_types_to_filter is None:
        entity_types_to_filter = ['APPLICANT', 'FUNDING_BODY', 'PERSON', 'LOCATION']

    start_time = time()
    print('\nFiltering invalid entity values from the graph database ...')

    with open(input_path, 'r', encoding='utf-8') as f:
        triplets = [json.loads(line) for line in f]
    print(f'Loaded {len(triplets)} triplets from {input_path}.')

    # Compile patterns that indicate a value is structurally not a named entity
    _months = (
        r'Jan(?:uary|uar)?|Feb(?:ruary|ruar)?|Mar(?:ch)?|März?|'
        r'Apr(?:il)?|May|Mai|Jun(?:e|i)?|Jul(?:y|i)?|Aug(?:ust)?|'
        r'Sep(?:tember)?|Oct(?:ober)?|Okt(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?|Dez(?:ember)?'
    )
    invalid_patterns = [
        re.compile(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$'),                        # German numeric date: 21.03.2025
        re.compile(r'^\d{4}-\d{2}-\d{2}'),                                  # ISO date: 2025-03-21
        re.compile(rf'\b(?:{_months})\b.{{0,20}}\b(?:19|20)\d{{2}}\b', re.IGNORECASE),  # Written-out date: "December 31, 2021" / "31. März 2025"
        re.compile(r'^[\d\s.,;:%()/\-+]+$'),                                # Purely numeric / formula
        re.compile(r'https?://', re.IGNORECASE),                            # URL with protocol
        re.compile(r'\bwww\.', re.IGNORECASE),                              # URL without protocol
        re.compile(r'[\w.\-]+@[\w.\-]+\.\w+'),                              # E-mail address
    ]

    def is_invalid(entity_type: str, value: str) -> bool:
        if entity_type not in entity_types_to_filter:
            return False
        if len(value.split()) > max_words:
            return True
        for pattern in invalid_patterns:
            if pattern.search(value):
                return True
        return False

    kept = []
    rejected = []
    rejection_counts: Dict[str, int] = {}

    for triplet in triplets:
        subj_invalid = is_invalid(triplet['subject_type'], triplet['subject_value'])
        obj_invalid = is_invalid(triplet['object_type'], triplet['object_value'])

        if subj_invalid or obj_invalid:
            rejected.append(triplet)
            if subj_invalid:
                rejection_counts[triplet['subject_type']] = rejection_counts.get(triplet['subject_type'], 0) + 1
            if obj_invalid:
                rejection_counts[triplet['object_type']] = rejection_counts.get(triplet['object_type'], 0) + 1
        else:
            kept.append(triplet)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for triplet in kept:
            f_out.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    with open(rejected_output_path, 'w', encoding='utf-8') as f_out:
        for triplet in rejected:
            f_out.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    end_time = time()
    pct = 100 * len(rejected) / len(triplets) if triplets else 0
    print(f'Kept {len(kept)} triplets, rejected {len(rejected)} ({pct:.1f}%).')
    if rejection_counts:
        print('  Rejections by entity type:')
        for t, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f'    {t}: {count}')
    print(f'Rejected triplets saved to {rejected_output_path} for inspection.')
    print(f'Filtered graph database saved to {output_path} in {end_time - start_time:.2f} seconds.\n')


### Helper function that converts the list of topics into a one-hot vector representation so that it can be added to the ChromaDB metadata
# Input:
# - [List[str]] topic_list: a list of the topics associated to a given text chunk (can be empty)
# Output:
# - [Dict] output_dict: a dictionary with keys corresponding to each topic, and value 0 or 1 depending on whether the topic is associated to the chunk
def get_topic_dict(topic_list: List[str]) -> Dict:
    output_dict = {
        'objective': 0,
        'inclusion_criteria': 0,
        'exclusion_criteria': 0,
        'deadline': 0,
        'max_funding': 0,
        'max_duration': 0,
        'procedure': 0,
        'contact': 0,
        'misc': 0
    }
    for topic in topic_list:
        output_dict[topic] = 1
    return output_dict


### Function that builds a ChromaDB structured database from the structured database
# Input:
# - [str] database_path: path to the jsonl file containing the structured database of chunks
# - [str] chroma_db_path: path to the ChromaDB database folder where the structured database should be saved
# - [str] embedding_model: name of the embedding model to be used for ChromaDB
# - [int] batch_size: number of entries to process in each batch
# Output:
# - None
def build_chromadb(
    database_path: str, 
    chroma_db_path: str, 
    embedding_model_path: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    batch_size: int = 500
) -> None:
    start_time = time()
    print('\nBuilding the ChromaDB structured database from the structured database ...')

    # Create the ChromaDB client and define the embedding function
    client = chromadb.PersistentClient(path=chroma_db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_path)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name="bekanntmachungen_index", embedding_function=embedding_func)

    # --- Intelligent resume logic: Check for existing IDs ---
    # Get the count of items already in the collection to fetch their IDs
    existing_count = collection.count()
    existing_ids = set()
    if existing_count > 0:
        print(f"Collection already contains {existing_count} entries. Fetching existing IDs to avoid duplicates.")
        # Fetch existing entries in batches to manage memory usage
        for offset in range(0, existing_count, batch_size):
            ids_batch = collection.get(limit=batch_size, offset=offset)['ids']
            existing_ids.update(ids_batch)
        print(f"Found {len(existing_ids)} unique IDs in the database.")

    # Load the source jsonl file and filter out entries that already exist
    with open(database_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    data_to_add = [entry for entry in all_data if entry['metadata']['id'] not in existing_ids]

    if not data_to_add:
        print("No new data to add. The ChromaDB collection is already up-to-date.")
        end_time = time()
        print(f"ChromaDB build process completed in {end_time - start_time:.2f} seconds.\n")
        return

    print(f"Found {len(data_to_add)} new entries to add to the database.")

    # --- Process and add data in batches to save progress incrementally ---
    total_batches = (len(data_to_add) + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch_start_time = time()
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_data = data_to_add[start_index:end_index]

        print(f"Processing batch {i+1}/{total_batches} ({len(batch_data)} entries)...")

        # Prepare the data for the current batch
        texts = [f"Document Type: {entry['metadata']['document_type']}. Section Title: {entry['metadata']['section_title']}. Text: {entry['text']}" for entry in batch_data]
        #metadatas = [entry['metadata'] for entry in batch_data]
        # Get the list of metadata with the list of topics converted into dictionary
        metadatas = []
        for entry in batch_data:
            metadata_tmp = entry['metadata']
            topics = metadata_tmp.pop('topics', None)
            entry['metadata'] = {**metadata_tmp, **get_topic_dict(topics)}
            metadatas.append(entry['metadata'])
        # Get the list of IDs
        ids = [entry['metadata']['id'] for entry in batch_data]

        # Add the batch to the collection
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        batch_end_time = time()
        print(f"  - Batch {i+1} added in {batch_end_time - batch_start_time:.2f} seconds.")

    end_time = time()
    print(f"ChromaDB structured database completed in {end_time - start_time:.2f} seconds.\n")


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

    ### Add the topics tags to the structured database of chunks
    if run_steps.get('add_topic_tags', False):
        print("\n--- Adding the 'topics' metadata tags to the entries of the structured database ---")
        parameters = config.get('add_topic_tags',{})
        assign_topic_to_chunk(
            database_path=parameters.get('database_path'),
            output_path=parameters.get('output_path')
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
            chunk_batch=parameters.get('chunk_batch',4),
            temperature=parameters.get('temperature',0.1)
        )

    ### Split conjunction triplets in the graph database
    if run_steps.get('split_graph_database', False):
        print("\n--- Splitting conjunction triplets in the graph database ---")
        parameters = config.get('split_graph_database', {})
        split_conjunction_triplets(
            input_path=parameters.get('input_path'),
            output_path=parameters.get('output_path'),
            entity_types_to_split=parameters.get('entity_types_to_split', None),
            conjunction_markers=parameters.get('conjunction_markers', None),
            min_part_words=parameters.get('min_part_words', 2)
        )

    ### Canonicalise entity values in the graph database
    if run_steps.get('canonicalise_graph_database', False):
        print("\n--- Canonicalising entity values in the graph database ---")
        parameters = config.get('canonicalise_graph_database', {})
        canonicalise_graph_database(
            input_path=parameters.get('input_path'),
            output_path=parameters.get('output_path'),
            mapping_output_path=parameters.get('mapping_output_path'),
            embedding_model_path=parameters.get('embedding_model_path', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
            similarity_threshold=parameters.get('similarity_threshold', 0.75),
            token_overlap_threshold=parameters.get('token_overlap_threshold', 0.5),
            department_separators=parameters.get('department_separators', None),
            conjunction_markers=parameters.get('conjunction_markers', None),
            entity_types_normalise_only=parameters.get('entity_types_normalise_only', None),
            entity_types_to_canonicalise=parameters.get('entity_types_to_canonicalise', None)
        )

    ### Filter invalid entity values from the graph database
    if run_steps.get('filter_graph_database', False):
        print("\n--- Filtering invalid entity values from the graph database ---")
        parameters = config.get('filter_graph_database', {})
        filter_graph_database(
            input_path=parameters.get('input_path'),
            output_path=parameters.get('output_path'),
            rejected_output_path=parameters.get('rejected_output_path'),
            entity_types_to_filter=parameters.get('entity_types_to_filter', None),
            max_words=parameters.get('max_words', 20)
        )

    ### Build the ChromaDB structured database for the graph RAG
    if run_steps.get('build_chromadb', False):
        print("\n--- Building the ChromaDB structured database for Graph RAG ---")
        parameters = config.get('build_chromadb',{}) 
        build_chromadb(
            database_path=parameters.get('database_path'),
            chroma_db_path=parameters.get('chroma_db_path'),
            embedding_model_path=parameters.get('embedding_model_path','sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
            batch_size=parameters.get('batch_size',500)
        )