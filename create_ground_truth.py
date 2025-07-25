
import os
import json
import random
import numpy as np
import shutil
from translate import Translator
from dotenv import load_dotenv
import google.generativeai as genai
from time import time
from pdb import set_trace as st


### Helper function to create the ground truth for a chosen subsample of the dataset.
### In order not to skew evaluation, amendments to the Bekanntmachungen are left out from the evaluation set
# Inputs:
# - [str] data_folder_path: path to the folder containing the Bekanntmachungen in text format
# - [str] result_folder_path: path to the root folder where to save the data files (both .txt and .json) of the evaluation set
# - [int] nb_samples: number of samples to select for the evaluation
# - [bool] translate: save a translated version of the Bekanntmachungen to make annotation easier.
# - [int] random_seed: parameter to fix the random seed to ensure the same evaluation subset is kept
def create_evaluation_set(data_folder_path, result_folder_path, nb_samples, translate=True, random_seed=12345):

    start = time()

    # Fix the random seed
    random.seed(random_seed)

    # List the text data files excluding amendments to the Bekanntmachungen
    file_list = [e for e in os.listdir(data_folder_path) if not 'Aenderung' in e and not 'aenderung' in e and not '%C3%84nderung' in e and not 'korrektur' in e] 
    
    # Shuffle the list and keep only the first nb_samples examples
    file_list = [file_list[i] for i in np.random.permutation(len(file_list))]
    kept_files = file_list[:nb_samples]

    # Define a template json file to save for annotation
    # NOTE: also include fields to keep track of the original document source span(s) for easier information verification
    json_result = {
        "objective": "",
        "objective_source_spans": "",
        "inclusion_criteria": "",
        "inclusion_criteria_source_spans": "",
        "exclusion_criteria": "",
        "exclusion_criteria_source_spans": "",
        "deadline": "",
        "deadline_source_spans": "",
        "max_funding": "",
        "max_funding_source_spans": "",
        "max_duration": "",
        "max_duration_source_spans": "",
        "procedure": "",
        "procedure_source_spans": "",
        "contact": "",
        "contact_source_spans": "",
        "misc": "",
        "misc_source_spans": ""
    }

    # Create and empty the result folders as necessary
    tmp_dir = os.path.join(result_folder_path,'data')
    if not os.path.isdir(tmp_dir):
        print("Creating data folder %s" % (tmp_dir))
        os.makedirs(tmp_dir)
    tmp_list = os.listdir(tmp_dir)
    if len(tmp_list)>0:
        print('Output directory %s non-empty: removing existing %d file(s)' % (tmp_dir, len(tmp_list)))
        for idx in range(len(tmp_list)):
            os.remove(os.path.join(tmp_dir,tmp_list[idx]))
    
    tmp_dir = os.path.join(result_folder_path,'ground_truth')
    if not os.path.isdir(tmp_dir):
        print("Creating ground truth folder %s" % (tmp_dir))
        os.makedirs(tmp_dir)
    tmp_list = os.listdir(tmp_dir)
    if len(tmp_list)>0:
        print('Output directory %s non-empty: removing existing %d file(s)' % (tmp_dir, len(tmp_list)))
        for idx in range(len(tmp_list)):
            os.remove(os.path.join(tmp_dir,tmp_list[idx]))

    if translate:
        tmp_dir = os.path.join(result_folder_path,'translated_data')
        if not os.path.isdir(tmp_dir):
            print("Creating translation folder %s" % (tmp_dir))
            os.makedirs(tmp_dir)
        tmp_list = os.listdir(tmp_dir)
        if len(tmp_list)>0:
            print('Output directory %s non-empty: removing existing %d file(s)' % (tmp_dir, len(tmp_list)))
            for idx in range(len(tmp_list)):
                os.remove(os.path.join(tmp_dir,tmp_list[idx]))

    # Load Gemini API key for translation
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY not found among the environment variables defined in .env")
    #os.environ['GOOGLE_API_KEY'] = gemini_api_key
    genai.configure(api_key=gemini_api_key)

    # Loop on the selected files
    for idx in range(nb_samples):
        print('Processing file %d/%d: %s' % (idx+1,nb_samples,kept_files[idx]))
        # Create a copy of the text file into the output directory
        shutil.copyfile(os.path.join(data_folder_path,kept_files[idx]),os.path.join(result_folder_path,'data/',kept_files[idx]))
        # Create an empty ground truth file to be filled in manually
        new_file_name = kept_files[idx].replace('.txt','.json')
        with open(os.path.join(result_folder_path,'ground_truth/',new_file_name),'w',encoding='utf-8') as f:
            json.dump(json_result,f,indent=4,ensure_ascii=False)
        # Create a translate copy of the file (if enabled)
        if translate:
            with open(os.path.join(data_folder_path,kept_files[idx]),'r',encoding='utf-8') as f:
                text_contents = f.read()

            # Perform the translation | NOTE: this attempt using the translate Python library is not efficient because of max token limitations that forces small chunking with possible loss of context.
            # translator = Translator(to_lang="en",from_lang="de")
            # translated_text = translator.translate(text_contents)
            
            # Alternative: use Gemini for translation
            llm = genai.GenerativeModel("gemini-2.5-flash") 
            prompt = f'Translate the following German text into English while keeping linebreaks (\\n): \n\n{text_contents}'
            
            try:
                translated_text = llm.generate_content(prompt).text # TODO: debug this as this can cause errors ( `Part`, but none were returned. Please check the `candidate.safety_ratings` to determine if the response was blocked.)
                # Save a copy of the translated files
                translated_file_name = kept_files[idx].replace('.txt','_EN.txt')
                with open(os.path.join(result_folder_path,'translated_data/',translated_file_name),'w',encoding='utf-8') as f:
                    f.write(translated_text)
            except Exception as e:
                print(f"The following error occurred for file {kept_files[idx]} during translation: {e}")

    end = time()
    print('Evaluation database created in %.2f seconds' % (end-start))


### Main function
if __name__ == "__main__":
    data_folder_path = './data/'
    result_folder_path = './evaluation/'
    nb_samples = 50
    create_evaluation_set(data_folder_path=data_folder_path,result_folder_path=result_folder_path,nb_samples=nb_samples)