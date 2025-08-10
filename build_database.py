import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
from time import sleep, time
import random  # For random sleep intervals
from typing import Optional
from pdb import set_trace as st

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


### Main function
if __name__ == "__main__":

    ### Meta parameters
    bmbf_base_url = "https://www.bmbf.de" # Base URL of the BMBF website
    bmbf_search_url = "https://www.bmbf.de/SiteGlobals/Forms/Suche/Bekanntmachungsuche/Bekanntmachungsuche_Formular.html?resourceId=1051612&pageLocale=de&input_=917290&gtp=1051594_list%253D1&resultsPerPage=50" # URL of the page listing Bekanntmachungen
    url_file = './meta_data/bekanntmachung_links.txt' # File
    output_dir = './data/' # Directory where the extracted text files will be saved
    min_delay_seconds = 1 # Minimum and maximum delay in seconds between requests not to overload the server
    max_delay_seconds = 1.5
    log_file = './meta_data/failed_content_urls.txt' # File where logs of failed URLs will be saved
    log_file2 = './meta_data/failed_content_urls2.txt' # File where logs of failed URLs will be saved (when running again the script onto the log files)

    ### Building the list of Bekanntmachung links
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get the links
    bekanntmachung_urls = get_bekanntmachung_links(bmbf_base_url, bmbf_search_url)
    print(f"Found {len(bekanntmachung_urls)} unique Bekanntmachung links.")
    # Save the list of links to a text file
    with open(url_file, 'w', encoding='utf-8') as f:
        for url in bekanntmachung_urls:
            f.write(url + '\n')

    ### Retrive the contents of each Bekanntmachung and save them in text format
    scrape_bekanntmachungen_content(url_file, output_dir, min_delay_seconds=min_delay_seconds, max_delay_seconds=max_delay_seconds, log_file=log_file)

    ### NOTE: Debugging purposes
    #scrape_bekanntmachungen_content(url_file, output_dir='./data_debug/', min_delay_seconds=min_delay_seconds, max_delay_seconds=max_delay_seconds, log_file=log_file, debug=20)

    ### Retry failed URLs from the log file
    scrape_bekanntmachungen_content(log_file, output_dir, min_delay_seconds=min_delay_seconds, max_delay_seconds=max_delay_seconds, log_file=log_file2)