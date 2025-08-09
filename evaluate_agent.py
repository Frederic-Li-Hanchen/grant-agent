import evaluate
import os
import bert_score
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import ChatPromptTemplate
import dotenv
from typing import List
from pdb import set_trace as st # For debugging purposes


### Function to compute the BLEU, ROUGE and BertScore metrics given a ground truth and generated output, and save them in a csv file.
# Input arguments:
# - [str] gt_path: path to the folder containing the ground truth json files
# - [str] gen_path: path to the folder containing the generated json files
# - [str] output_filepath: path to save the computed metrics in csv format
def compute_metrics(gt_path, gen_path, output_filepath):
    """
    Compute BLEU, ROUGE, and BertScore metrics between ground truth and generated outputs.
    
    Args:
        gt_path (str): Path to the folder containing ground truth outputs.
        gen_path (str): Path to the folder containing generated outputs.
        output_filepath (str): Path to save the computed metrics in CSV format.
    """
    # Ensure paths exist
    if not os.path.exists(gt_path) or not os.path.exists(gen_path):
        raise FileNotFoundError("One of the specified paths does not exist.")

    # Load ground truth and generated outputs
    # NOTE: ground truth and generated files corresponding to the same sample are expected to have the same name
    gt_files = sorted(os.listdir(gt_path))
    gen_files = sorted(os.listdir(gen_path))

    # Keep only the files that are present in both directories
    if len(gt_files) != len(gen_files):
        print(f"WARNING: different number of ground truth ({len(gt_files)}) and generated ({len(gen_files)}) files. Keeping only matching files.")
        gt_files = [f for f in gt_files if f in gen_files]
        gen_files = [f for f in gen_files if f in gt_files]
    if not gt_files:
        print("ERROR: No matching files found between ground truth and generated outputs.")
        return

    # List of keys to extract from the json files
    json_keys = ["objective", "inclusion_criteria", "exclusion_criteria", "deadline", "max_funding", "max_duration", "procedure", "contact", "misc"]
    results = []

    # Load metric calculators once for efficiency
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # Loop on the files
    for gt_file, gen_file in zip(gt_files, gen_files):
        # Check if the files match
        if gt_file != gen_file:
            print(f"Warning: File names do not match, skipping: {gt_file} vs {gen_file}")
            continue

        print(f"\nProcessing file {gt_file} ...")
        try:
            with open(os.path.join(gt_path, gt_file), 'r', encoding='utf-8') as f:
                gt_json = json.load(f)
            with open(os.path.join(gen_path, gen_file), 'r', encoding='utf-8') as f:
                gen_json = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error reading JSON for {gt_file}: {e}. Skipping.")
            continue

        # Loop on the keys
        for key in json_keys:
            prediction = gen_json.get(key, "")
            reference = gt_json.get(key, "")

            if not isinstance(prediction, str) or not isinstance(reference, str):
                print(f"Warning: Non-string value for key '{key}' in file {gt_file}. Skipping metric calculation for this key.")
                continue

            # Apply preprocessing on the generated outputs
            prediction = prediction.replace('\n', ' ').replace('**', '').strip() # Remove newlines, asterisks and leading/trailing whitespaces
            prediction = re.sub(r'\s+', ' ', prediction) # Collapse multiple spaces into one
            prediction = prediction.replace("Not specified.", "Not specified").strip() # Replace "Not specified." with "Not specified"

            # Compute BLEU score| NOTE: for single words, BLEU will return 0 even if the generated answer matches the ground truth
            bleu_result = bleu_metric.compute(predictions=[prediction], references=[[reference]])
            assert bleu_result is not None, f"BLEU score is None for {gt_file} on key {key}"
            results.append({'file': gt_file, 'field': key, 'metric': 'BLEU', 'score': bleu_result['bleu']})

            # Compute ROUGE score
            rouge_result = rouge_metric.compute(predictions=[prediction], references=[[reference]])
            assert rouge_result is not None, f"ROUGE score is None for {gt_file} on key {key}"
            results.append({'file': gt_file, 'field': key, 'metric': 'ROUGE-1', 'score': rouge_result['rouge1']})

            # Compute BertScore
            # bert_score.score returns (precision, recall, f1)
            _, _, bert_f1 = bert_score.score([prediction], [[reference]], lang='en', verbose=False)
            results.append({'file': gt_file, 'field': key, 'metric': 'BertScore_F1', 'score': bert_f1.item()})

    if results:
        metrics_df = pd.DataFrame(results)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        metrics_df.to_csv(output_filepath, index=False)
        print(f"\nMetrics saved to {output_filepath}")


### Function to plot boxplots and violin plots of computed metrics given a csv file containing them
# Input arguments:
# - [str] csv_filepath: path to the csv file containing the computed metrics
# - [str] output_folder_path: path to the folder where to save both box and violin plots
def plot_metrics(csv_filepath: str, output_folder_path: str):
    """
    Plot grouped boxplots and violin plots of computed metrics from a CSV file.
    
    For each JSON field, a group of plots (one for each metric) is created.
    
    Args:
        csv_filepath (str): Path to the CSV file containing metrics.
        output_folder_path (str): Path to save the boxplot image. A violin plot
            is also saved to a derived path.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file {csv_filepath} does not exist.")

    # Load the metrics from the CSV file
    metrics_df = pd.read_csv(csv_filepath)

    # Define the order of metrics and fields for consistent plotting
    metric_order = ['BLEU', 'ROUGE-1', 'BertScore_F1']
    field_order = ["objective", "inclusion_criteria", "exclusion_criteria", "deadline", 
                   "max_funding", "max_duration", "procedure", "contact", "misc"]
    
    # Filter the DataFrame to only include the metrics we want to plot
    # and ensure the 'metric' column is categorical to respect the order
    metrics_df = metrics_df[metrics_df['metric'].isin(metric_order)].copy()
    if metrics_df.empty:
        print("No metrics found in the CSV to plot.")
        return
    metrics_df['metric'] = pd.Categorical(metrics_df['metric'], categories=metric_order, ordered=True)

    # Define colors for each metric
    colors = {'BLEU': 'blue', 'ROUGE-1': 'red', 'BertScore_F1': 'green'}

    # --- Create Box Plot ---
    fig, ax = plt.subplots(figsize=(20, 10))
    
    sns.boxplot(
        data=metrics_df,
        x='field', 
        y='score', 
        hue='metric',
        order=field_order,
        hue_order=metric_order,
        palette=colors,
        ax=ax
    )
    
    # Set plot titles and labels
    ax.set_title('Metric Score Distributions by Field', fontsize=16, pad=15)
    ax.set_xlabel('Field', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Improve legend
    ax.legend(title='Metric')
    
    # Adjust layout to prevent labels overlapping
    fig.tight_layout()

    # Save the plot
    # output_dir = os.path.dirname(output_filepath)
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)
    if output_folder_path:
        os.makedirs(output_folder_path, exist_ok=True)
        
    # plt.savefig(output_filepath, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder_path, 'boxplot.png'), bbox_inches='tight')
    print(f"Box plot saved to {output_folder_path}")
    plt.close(fig) # Close the figure to free memory

    # --- Create Violin Plot ---
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.violinplot(
        data=metrics_df,
        x='field',
        y='score',
        hue='metric',
        order=field_order,
        hue_order=metric_order,
        palette=colors,
        ax=ax
    )

    # Set plot titles and labels
    ax.set_title('Metric Score Distributions by Field', fontsize=16, pad=15)
    ax.set_xlabel('Field', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Improve legend
    ax.legend(title='Metric')

    # Adjust layout to prevent labels overlapping
    fig.tight_layout()

    # # Save the plot to a derived filepath
    plt.savefig(os.path.join(output_folder_path,'violinplot.png'), bbox_inches='tight')
    print(f"Violin plot saved to {output_folder_path}")
    plt.close(fig)  # Close the figure to free memory


### Helper function that given a ground truth json, specific json field and associated .txt file, returns the text sections from the file where the information to annotated the field is located
# Input arguments:
# - [str] doc_path: path to the text document containing the German call
# - [List[str]] source_spans: a list of section identifiers to extract, e.g. ["1", "annex 3", "introduction"]
# Returns:
# - [str] a string containing the concatenated text of the requested sections, separated by "\n\n\n\n\n". Returns the full document if source_spans is empty or contains "document"
def get_relevant_text_sections(doc_path: str, source_spans: List[str]) -> str:
    """
    Extracts specific sections from a grant document based on section numbers or keywords.

    All German documents follow a similar structure, with sections indicated by a
    number (e.g., "7.2.1 Section title\\n").
    If source_spans is empty or contains "document", the whole document is returned.
    source_spans can also contain (not case-sensitive) "Annex n" for an appendix
    ("Anlage"), or "Introduction" for the part of the document before section 1.

    Args:
        doc_path: The path to the text document containing the German call.
        source_spans: A list of section identifiers to extract.

    Returns:
        A string containing the concatenated text of the requested sections,
        separated by "\\n\\n\\n\\n\\n". Returns the full document if source_spans is
        empty or contains "document".
    """
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        return f"Error: Document not found at {doc_path}"

    # If source_spans is empty or requests the whole document, return everything.
    if not source_spans or any(span.lower().strip() == 'document' for span in source_spans):
        return full_text

    # Regex to find section numbers like 1., 1.1, 2.3.4 etc. at the start of a line.
    section_pattern = re.compile(r'^\s*(\d(?:\.\d){0,2})\.?\s+.*')
    # Regex for Annexes (Anlage)
    annex_header_pattern = re.compile(r'^\s*Anlage\s*$', re.IGNORECASE)  # Matches lines with only "Anlage"

    # Collect both sections and annexes
    sections = {}
    annexes = {}
    inside_annex = False
    current_section_number = 'introduction'
    current_annex_number = 'introduction'
    current_content = []

    lines = full_text.split('\n')

    # Loop on the lines of the document
    for line in lines:
        if annex_header_pattern.match(line): # If the keyword "Anlage" is found alone, bordered by line breaks, this indicates that the next sections belong to an annex
            inside_annex = True
            sections[current_section_number] = '\n'.join(current_content).strip()
            current_content = [line]
            continue
        section_match = section_pattern.match(line)
        
        # Determine if the line is a new section or annex header
        new_section_number = None
        if section_match:
            new_section_number = section_match.group(1)
       
        if new_section_number:
            # Store the content of the previous section
            if current_content:
                if inside_annex:
                    annexes[current_annex_number] = '\n'.join(current_content).strip()
                else:
                    sections[current_section_number] = '\n'.join(current_content).strip()
            
            # Start the new section
            if inside_annex:
                current_annex_number = new_section_number
            else:
                current_section_number = new_section_number
            current_content = [line]
        else:
            current_content.append(line)

    # Store the last section after the loop finishes
    if current_content:
        if inside_annex:
            annexes[current_annex_number] = '\n'.join(current_content).strip()
        else:
            sections[current_section_number] = '\n'.join(current_content).strip()

    # Extract and concatenate the requested sections
    extracted_texts = []
    for span in set(s.lower().strip() for s in source_spans):
        if span == 'introduction':
            if 'introduction' in sections:
                extracted_texts.append(sections['introduction'])
        elif span.startswith('annex'):
            # Extract the annex number (e.g. "1" for "annex 1")
            annex_number = span.split('annex')[-1].strip()
            if annex_number in annexes:
                extracted_texts.append(annexes[annex_number])
        else:  # It's a section number
            # Find the content for the exact section number.
            # This prevents including subsections (e.g., "1.1" when "1" is requested).
            if span in sections:
                extracted_texts.append(sections[span])

    return "\n\n\n\n\n ".join(extracted_texts)


### Function that implements LLM-as-a-judge evaluation given a ground truth and generated output in csv format
# NOTE: work in progress
def llm_as_a_judge_evaluation(gt_path: str, gen_path: str, output_filepath: str, llm_model: str = "gpt-4o"):

    # Ensure paths exist
    if not os.path.exists(gt_path) or not os.path.exists(gen_path):
        raise FileNotFoundError("One of the specified paths does not exist.")

    # Load ground truth and generated outputs
    # NOTE: ground truth and generated files corresponding to the same sample are expected to have the same name
    gt_files = sorted(os.listdir(gt_path))
    gen_files = sorted(os.listdir(gen_path))

    # Keep only the files that are present in both directories
    if len(gt_files) != len(gen_files):
        print(f"WARNING: different number of ground truth ({len(gt_files)}) and generated ({len(gen_files)}) files. Keeping only matching files.")
        gt_files = [f for f in gt_files if f in gen_files]
        gen_files = [f for f in gen_files if f in gt_files]
    if not gt_files:
        print("ERROR: No matching files found between ground truth and generated outputs.")
        return

    # List of keys to extract from the json files
    json_keys = ["objective", "inclusion_criteria", "exclusion_criteria", "deadline", "max_funding", "max_duration", "procedure", "contact", "misc"]
    results = []

    # Define the Langchain prompt template that include all evaluation criteria to be scored by the LLM
    # TODO: the prompt should contain modular fields to be defined depending on the json field being evaluated
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert in evaluating information extraction tasks from grant documents.\n\
         You will be provided with some specific information to extract, relevant parts of the original grant document (in German) where the information can be found, the output generated by an agent, the ground truth, some formatting instructions and some examples of scoring.\n\
         Given all aforementioned information, provide a score on a 5-point Likert scale between 1 (very poor) and 5 (perfect) for each of the following evaluation criterion:\n\
         ### Criteria:\n\
         1. **Correctness**: is the generated output factually true and verifiable from the original grant document?\n\
         2. **Completeness**: does the generated output contain all elements relevant to the specific information to extract?\n\
         3. **Clarity**: is the language of the generated output clear and easy to understand?\n\
         4. **Conciseness**: is the generated output to the point, without unnecessary repetitions or irrelevant information?\n\
         5. **Adherence**: does the generated output follow the formatting instructions provided?\n\
        Return your scores in a json format where each of the five criteria is a key, and the score the associated value.\n\
        For example: {'correctness': 4, 'completeness': 3, 'clarity': 5, 'conciseness': 2, 'adherence': 1}\n"), # NOTE: if I want the explanations, I need to add a new json field that would contain them here.
        ("user", 
         "### Information to extract: {json_field}\n\
         ### Extracts from the grant document (in German): {relevant_parts}\n\
         ### Generated output: {generated_output}\n\
         ### Ground truth: {ground_truth}\n\
         ### Formatting instructions: {formatting_instructions}\n\
         ### Examples: {examples}")
    ])

    formatting_instructions = {key: "" for key in json_keys} # TODO: fill this dictionary with specific instructions for each json field
    examples = {key: "" for key in json_keys} # TODO: fill this dictionary with specific examples for each json field

    # Initialise the LLM client

    # Loop on the files to get the scores for each evaluation criterion
    for gt_file, gen_file in zip(gt_files, gen_files):
        pass

    # Save the scores to a CSV file


### Main function
if __name__ == '__main__':

    # Compute BLEU, ROUGE and BertScore metrics
    # gt_path = r'.\evaluation\ground_truth'
    # gen_path = r'.\evaluation\generated_outputs\gemini-2.5-flash'
    output_filepath = r'.\evaluation\metrics\gemini-2.5-flash\metrics.csv'
    # compute_metrics(gt_path, gen_path, output_filepath)

    # Plot boxplots and violin plots of metrics per json field
    output_folder_path = r'.\evaluation\plots\gemini-2.5-flash'
    plot_metrics(output_filepath, output_folder_path)