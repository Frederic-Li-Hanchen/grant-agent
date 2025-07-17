from pdb import set_trace as st
import re
import os

### Function to check whether the text files are properly formatted, by performing tests on patterns for which normalisation was required
# Input:
# - [str] database_path: the path to the directory that contains the text files
# - [str] log_file_path: the path to the directory that contains the log file to save
def check_database_quality(database_path, log_file_path):
   
    # Various meta parameters
    nb_amended_bekanntmachungen = 0 # Number of amended Bekanntmachungen
    files_with_improper_formatting = {'url': [], 'unicode': [], 'percentage': [], 'acronym': []} # Dictionary to store the path of files with improper formatting
    acronym_list = ["GmbH", "GbR", "kW", "mRNA", "WiFi", "UVLicht", "h2o", "H2O", "CO2", "co2", "IoT"] # List of commonly encountered acronyms | NOTE: could create false positives (e.g. BIO 2 triggers O2 warning)
    unicode_expression = rf'\xad|[\u2000-\u200A\u202F\u3000]|\xa0|[\u200B-\u200F\uFEFF]|[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u0080-\u009F]|[\u2080-\u2089]|[\u00B9\u00B2\u00B3\u2070\u2074-\u2079]|[\uE000-\uF8FF\uFDD0-\uFDEF]'

    # Loop on the text files
    nb_files = len(os.listdir(database_path))
    print('Found %d text files to check' % nb_files)

    for filename in os.listdir(database_path):
        if filename.endswith('.txt'):
            if 'aenderung' in filename or 'Aenderung' in filename:
                nb_amended_bekanntmachungen += 1
            file_path = os.path.join(database_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f'Checking file: {filename}')

            # Web links and email addresses: check that they are properly separated from the text before and after
            # TODO: currently can raise false positives for properly formatted links (e.g. "some text (other text http://some-url.com/abc/def)") -> relax the checking rule
            url_found = re.findall(r'(\[https?://[^\s]+\]|\[http?://[^\s]+\]|\(https?://[^\s]+\)|\(http?://[^\s]+\)|https?://[^\s]+|http?://[^\s]+|\(www\.[^\s]+\)+|\[www\.[^\s]+\]+|www\.[^\s]+|[\w\.-]+@[\w\.-]+\.\w+)', text)
            if url_found:
                for url in url_found:
                    # if ')' in url and not '(' in url:
                    #     st()
                    if not re.search(rf'(\s\({re.escape(url)}\)\s|\s\[{re.escape(url)}\]\s|\s{re.escape(url)}\s)', text):
                        print(f'  - URL or email "{url}" not properly separated in {filename}')
                        files_with_improper_formatting['url'].append((filename,url))

            # Percentages: check that there is always a space between numbers and percentages for consistency (except in URLs)
            percentage_found = re.findall(r'\d+%|\d+\s*%|\d Percent|\d+\s*percent|\d Prozent|\d+\s*prozent', text)
            if percentage_found:
                for percentage in percentage_found:
                    if not re.search(r'\d+\s+(%|percent|Percent|Prozent|prozent)', percentage):
                        print(f'  - Percentage "{percentage}" not properly separated in {filename}')
                        files_with_improper_formatting['percentage'].append((filename,percentage))

            # Acronyms: check that the acronyms defined in the provided list are not broken down by spaces
            # NOTE: may raise some false positives (e.g. "BIO 2" -> "O2")
            for acronym in acronym_list:
                # Pattern to match acronym with optional spaces between letters (e.g. G m b H)
                broken_pattern = r'\s*'.join(list(acronym))
                if re.search(broken_pattern, text) and not re.search(rf'\b{acronym}\b', text):
                    print(f'  - Acronym "{acronym}" is broken in {filename}')
                    files_with_improper_formatting['acronym'].append((filename, acronym))

            # # Unicode: check that there are no leftover unicode characters
            # NOTE: disabled because returns many false positives?
            # if re.search(unicode_expression, text):
            #     print(f'  - Unwanted unicode character found in {filename}')
            #     files_with_improper_formatting['unicode'].append(filename)
                             
    # Save the contents of the dictionary into a log file
    with open(log_file_path, 'w', encoding='utf-8') as f:
        for key in files_with_improper_formatting.keys():
            f.write(f"### Files containing issues with {key} ###\n")
            for item in files_with_improper_formatting[key]:
                f.write(f"- {item[0]}: {item[1]}\n")
            f.write('\n')

    print(f"Improperly formatted files saved to {log_file_path}")
    print(f"{nb_amended_bekanntmachungen} amended Bekanntmachungen found out of {nb_files}")



### Main
if __name__ == '__main__':
    database_path = './data/'
    log_file_path = './meta_data/improper_formatting_log.txt'
    check_database_quality(database_path=database_path,log_file_path=log_file_path)