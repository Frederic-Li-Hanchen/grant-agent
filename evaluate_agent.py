import evaluate
import os
import bert_score
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
#from langchain.chains import SimpleSequentialChain, LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
#from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#import tenacity
from langchain_groq import ChatGroq
import time
import requests
import groq
import csv
import sys
import numpy as np
from utils import spearman_corr_custom
from utils import load_config_from_yaml
import torch
from BARTScore.bart_score import BARTScorer
from pdb import set_trace as st # For debugging purposes


### Function to compute the BLEU, ROUGE and BertScore metrics given a ground truth and generated output, and save them in a csv file.
# Input arguments:
# - [str] gt_path: path to the folder containing the ground truth json files
# - [str] gen_path: path to the folder containing the generated json files
# - [str] output_filepath: path to save the computed metrics in csv format
def compute_metrics(gt_path: str, gen_path: str, output_filepath: str) -> None:
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

    # Initialise the BARTScorer, dynamically selecting the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing BARTScorer on device: {device}")
    scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    scorer.load(path='BARTScore/bart_score.pth')

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

            # Compute BLEU score. Handle the edge case where the prediction is an empty
            # string, which would cause a ZeroDivisionError. In this case, the score is 0.
            if not prediction:
                bleu_result = {'bleu': 0.0}
            else:
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

            # Compute BartScore
            score1 = scorer.score(srcs=[reference], tgts=[prediction])[0]
            score2 = scorer.score(srcs=[prediction], tgts=[reference])[0]
            score = (score1+score2)/2
            results.append({'file': gt_file, 'field': key, 'metric': 'BartScore', 'score': score})

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
def plot_metrics(csv_filepath: str, output_folder_path: str) -> None:
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

    # Define the order of metrics and fields for consistent plotting.
    # BartScore is now included.
    metric_order = ['BLEU', 'ROUGE-1', 'BertScore_F1', 'BartScore']
    field_order = ["objective", "inclusion_criteria", "exclusion_criteria", "deadline", 
                   "max_funding", "max_duration", "procedure", "contact", "misc"]
    
    # Filter the DataFrame to only include the metrics we want to plot
    # and ensure the 'metric' column is categorical to respect the order
    metrics_df = metrics_df[metrics_df['metric'].isin(metric_order)].copy()

    if metrics_df.empty:
        print("No metrics found in the CSV to plot.")
        return
        
    metrics_df['metric'] = pd.Categorical(metrics_df['metric'], categories=metric_order, ordered=True)

    # Define a color mapping for the metrics
    palette = {'BLEU': 'blue', 'ROUGE-1': 'red', 'BertScore_F1': 'green', 'BartScore': 'purple'}

    # --- Create Faceted Box Plot ---
    # Use catplot to create faceted plots, with each metric in its own subplot.
    # This handles different y-axis scales gracefully.
    g = sns.catplot(
        data=metrics_df,
        x='field', 
        y='score', 
        col='metric',  # Create a column of subplots for each metric
        kind='box',    # Specify the plot kind
        order=field_order,
        col_wrap=2,    # Wrap the subplots into 2 columns
        sharex=True,   # Share the x-axis labels
        sharey=False,  # Each subplot has its own y-axis scale
        height=5, 
        aspect=1.5
    )
    
    # Customize titles and labels for each subplot
    g.fig.suptitle('Metric Score Distributions by Field', fontsize=16, y=1.03)
    g.set_axis_labels("Field", "Score")
    g.set_titles("Metric: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')
    g.tick_params(axis='x', labelsize=9)

    # Helper function to set all colors of a boxplot
    def set_boxplot_color(ax, color):
        # Set the color for the main box patches
        for box in ax.patches:
            box.set_facecolor(color)
        # # Set the color for whiskers, caps, and medians
        # for line in ax.lines:
        #     line.set_color(color)

    # Apply custom colors and y-axis limits to each subplot
    for ax in g.axes.flat:
        metric_name = ax.get_title().replace('Metric: ', '')
        color = palette.get(metric_name)
        if color:
            set_boxplot_color(ax, color)
        
        # The title of each subplot is 'Metric: {col_name}'
        if ax.get_title() in ['Metric: BLEU', 'Metric: ROUGE-1', 'Metric: BertScore_F1']:
            ax.set_ylim(-0.05, 1.05)

    g.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for labels and title

    # Save the plot
    if output_folder_path:
        os.makedirs(output_folder_path, exist_ok=True)
        boxplot_path = os.path.join(output_folder_path, 'metrics_boxplot_faceted.png')
        g.savefig(boxplot_path)
        print(f"Faceted box plot saved to {boxplot_path}")
    plt.close(g.fig)

    # --- Create Faceted Violin Plot ---
    g = sns.catplot(
        data=metrics_df, 
        x='field', 
        y='score', 
        col='metric', 
        kind='violin', 
        order=field_order, 
        col_wrap=2, 
        sharex=True, 
        sharey=False, 
        height=5, 
        aspect=1.5)
    g.fig.suptitle('Metric Score Distributions by Field', fontsize=16, y=1.03)
    g.set_axis_labels("Field", "Score")
    g.set_titles("Metric: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')
    g.tick_params(axis='x', labelsize=9)

    # Apply custom colors and y-axis limits to each subplot
    for ax in g.axes.flat:
        metric_name = ax.get_title().replace('Metric: ', '')
        color = palette.get(metric_name)
        if color:
            for collection in ax.collections:
                collection.set_facecolor(color)
        
        if ax.get_title() in ['Metric: BLEU', 'Metric: ROUGE-1', 'Metric: BertScore_F1']:
            ax.set_ylim(-0.05, 1.05)

    g.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for labels and title
    
    if output_folder_path:
        violinplot_path = os.path.join(output_folder_path, 'metrics_violinplot_faceted.png')
        g.savefig(violinplot_path)
        print(f"Faceted violin plot saved to {violinplot_path}")
    plt.close(g.fig)


### Helper function that given a path to text file and a list of source_spans, returns the text sections from the file where the information to annotated the field is located
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

    # If source_spans is requests the whole document, return everything.
    if any(span.lower().strip() == 'document' for span in source_spans):
        return full_text
    
    # If source_spans is empty or requests the whole document, return everything.
    if not source_spans or any(span.lower().strip() == 'document' for span in source_spans):
        return full_text
    
    # Regex to find section numbers like 1., 1.1, 2.3.4 etc. at the start of a line.
    # NOTE: some section numbers may be misformatted (e.g. 7.21 instead of 7.2.1), so this should be covered by the regex
    section_pattern = re.compile(r'^\s*(\d(?:\.\d){0,2})\.?\s+.*')
    # Regex for Annexes (Anlage)
    annex_header_pattern = re.compile(r'^\s*Anlage\s*$', re.IGNORECASE)  # Matches lines with only "Anlage"

    # Collect both sections and annexes
    sections = {}
    annexes = {}
    inside_annex = False
    current_section_number = 'introduction' # Tracks the current section being built
    current_annex_number = 'introduction'
    current_content = []

    lines = full_text.split('\n')

    # Loop on the lines of the document
    for line in lines:
        if annex_header_pattern.match(line): # If the keyword "Anlage" is found alone, bordered by line breaks, this indicates that the next sections belong to an annex
            inside_annex = True
            # Store the content of the previous section before starting the annex
            if current_content:
                sections[current_section_number] = '\n'.join(current_content).strip()
            current_section_number = 'annex_introduction' # Reset for content within the annex but before the first numbered section
            current_content = [line] # Start new content block with the "Anlage" line
            continue
        section_match = section_pattern.match(line)
        
        # Determine if the line is a new section or annex header
        new_section_number = None
        if section_match:
            new_section_number = section_match.group(1)
       
        if new_section_number:
            # Store the content of the previous section
            if current_content:
                # This logic assumes annexes appear after the main content.
                # If a main section number appears after an annex, it will be treated as part of the annex.
                if inside_annex:
                    # Use the new section number as the key for the annex
                    annexes[new_section_number] = '\n'.join(current_content).strip()
                else:
                    sections[current_section_number] = '\n'.join(current_content).strip()
            
            # Start the new section
            if inside_annex:
                current_annex_number = new_section_number
            else:
                # This logic correctly handles the transition from introduction to the first numbered section
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



### Function that implements LLM-as-a-judge evaluation given a ground truth and generated output in csv format.
### Relevant information is retrieved from the original documents via vector RAG to help with the evaluation.
# Input arguments:
# - [str] gt_path: path to the folder containing the ground truth json files
# - [str] gen_path: path to the folder containing the generated output in json format
# - [str] text_path: path to the folder containing the original text documents
# - [str] csv_output_filepath: path to the CSV file where results of the evaluation should be saved
# - [BaseLanguageModel] llm: Langchain LLM
# - [str] embedding_name: name of the Huggingface embedding model to use
# - [int] max_context_length: maximum length (in characters) of the context retrieved from the original document
# - [int] chunk_size: size (in characters) of the chunks to segment the original document
# - [int] chunk_overlap: overlap (in characters) between the segmented chunks of the original document
# - [int] top_k: number of most similar chunks to the query passed as input to the LLM judge
def llm_as_a_judge_evaluation(
        gt_path: str, 
        gen_path: str, 
        text_path: str, 
        csv_output_filepath: str, 
        llm: BaseLanguageModel,
        embedding_name: str="all-MiniLM-L6-v2",
        max_context_length: int=12000,
        chunk_size: int=1500,
        chunk_overlap: int=150,
        top_k: int=5,
        examples_json: str='./prompts/llm_judge_examples.json',
        formatting_instructions_json: str='./prompts/llm_judge_formatting_instructions.json') -> None:

    start = time.time()

    # Resume Logic: Check for existing results to avoid re-processing
    already_evaluated = set()
    existing_df = None
    existing_csv_lines = []
    if os.path.exists(csv_output_filepath):
        try:
            print(f"Found existing results file at {csv_output_filepath}. Will skip already evaluated fields.")
            existing_df = pd.read_csv(csv_output_filepath)
            for _, row in existing_df.iterrows():
                already_evaluated.add((row['file'], row['field']))
            # Load the already computed results from the csv file into a list of dictionaries
            with open(csv_output_filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_csv_lines = list(reader)
        except pd.errors.EmptyDataError:
            print("Existing results file is empty. Starting fresh.")
            existing_df = None
    else:
        # Make sure the directory where the csv file is to be saved exists
        os.makedirs(os.path.dirname(csv_output_filepath), exist_ok=True)

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

    # Define the output parser that also defines formatting instructions
    # Pydantic model for the LLM-as-a-judge output
    class JudgeOutput(BaseModel):
        scores: Dict[str, int] = Field(description="A dictionary where keys are the evaluation criteria (correctness, completeness, clarity, conciseness, adherence) and values are the scores from 1 to 5.")
        explanations: Dict[str, str] = Field(description="A dictionary where keys are the evaluation criteria and values are the textual explanations for the scores.")
    
    # Use a self-correcting parser that can fix malformed JSON from the LLM
    parser = PydanticOutputParser(pydantic_object=JudgeOutput)
    #output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # Define the Langchain prompt template that include all evaluation criteria to be scored by the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "### Instructions:\n\
         You are an expert in evaluating information extraction tasks from grant documents.\n\
         You will be provided with contextual information specifying information to extract, relevant parts of the original grant document (in German), \
         the output generated by an agent, the ground truth, some formatting instructions and some examples of scoring.\n\
         Given this context, provide a score on a 5-point Likert scale between 1 (very poor) and 5 (perfect) for each of the following evaluation criterion:\n\
         1. **Correctness:** is the generated output factually true and verifiable from the original grant document?\n\
         2. **Completeness:** does the generated output contain all elements relevant to the specific information to extract?\n\
         3. **Clarity:** is the language of the generated output clear and easy to understand?\n\
         4. **Conciseness:** is the generated output to the point, without unnecessary repetitions or irrelevant information?\n\
         5. **Adherence:** does the generated output follow the formatting instructions provided?\n\
        ### Evaluation rules:\n\
        1. Your main goal is to assess if the **generated output** correctly captures the information requested by each **criterion**, as reflected in the **ground truth**.\n\
        2. Crucially, if both the **ground truth** and the **generated output** are the same **including 'None' or 'Not specified'** (punctuation aside), it means they are in \
        perfect agreement that the information is not available in the context. In these cases, you MUST assign a score of 5 to all criteria, and explain that the agent correctly \
        identified that the information was not found in the document.\n\
        ### Output format instructions:\n\
        {output_format_instructions}"),
        ("user", 
         "### Information to extract: {json_field}\n\
         ### Specific instructions: {formatting_instructions}\n\
         ### Extracts from the grant document (in German): {relevant_parts}\n\
         ### Ground truth: {ground_truth}\n\
         ### Generated output: {generated_output}\n\
         ### Example(s): {examples}")
    ])

    # Load the json files used to define the prompt
    with open(formatting_instructions_json, 'r', encoding='utf-8') as f:
        formatting_instructions = json.load(f)

    with open(examples_json, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # Initialize the retriever components once per evaluation run
    embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Open the csv file of results
    with open(csv_output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'field', 'criterion', 'score', 'explanation'])
        writer.writeheader()
        writer.writerows(existing_csv_lines)

        # Loop on the files to get the scores for each evaluation criterion
        for gt_file, gen_file in zip(gt_files, gen_files):
            print(f"Evaluating {gt_file}...")

            # Construct path to the original document
            doc_name = gt_file.replace('.json', '.txt')
            doc_path = os.path.join(text_path, doc_name)

            if not os.path.exists(doc_path):
                print(f"  Warning: Could not find source document at {doc_path}. Skipping.")
                continue

            # Load JSON files
            try:
                with open(os.path.join(gt_path, gt_file), 'r', encoding='utf-8') as f:
                    gt_json = json.load(f)
                with open(os.path.join(gen_path, gen_file), 'r', encoding='utf-8') as f:
                    gen_json = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  Error reading JSON for {gt_file}: {e}. Skipping.")
                continue

            # Create a chain with a fallback for parsing errors.
            # If the initial parsing fails, it passes the output and the error to a fixing chain.
            fixing_prompt_template = PromptTemplate.from_template(
                "Fix the following output to conform to the format instructions. Do not add any other text.\n\n"
                "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
                "FAILED OUTPUT:\n{completion}\n\n"
                "ERROR:\n{error}"
            )
            fixing_chain = fixing_prompt_template | llm | parser
            chain = (prompt | llm | parser).with_fallbacks(
                fallbacks=[fixing_chain],
                exception_key="error",
                input_key="completion"
            ).with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True,
                retry_if_exception_type=(groq.APIStatusError, requests.exceptions.RequestException)
            )

            # Loop on the json fields to evaluate
            for key in json_keys:
                # Resume Logic: Skip if already evaluated
                if (gt_file, key) in already_evaluated:
                    print(f"  - Skipping already evaluated field: {key}")
                    continue

                print(f"  - Evaluating field: {key}")

                # Prepare the inputs for the prompt
                ground_truth = gt_json.get(key, "")
                generated_output = gen_json.get(key, "")

                # Get source spans and extract relevant text
                source_spans_key = f"{key}_source_spans"
                source_spans_str = gt_json.get(source_spans_key, "")
                source_spans = [span.strip() for span in source_spans_str]

                # If source spans are not defined (or refer to the whole document),
                # the context might be too long. Use RAG to find the most relevant parts.
                if not source_spans or any(span.lower().strip() == 'document' for span in source_spans):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()

                    if len(full_text) > max_context_length:
                        print(f"    - Context for '{key}' is the full document and is too long ({len(full_text)} chars). Using RAG to find relevant parts.")
                        
                        # 1. Create a targeted retrieval query based on the evaluation task
                        retrieval_query = f"Information about '{key}': {formatting_instructions.get(key, '')}"
                        
                        # 2. Split text and create an in-memory retriever
                        docs = text_splitter.create_documents([full_text])
                        vector_store = FAISS.from_documents(docs, embeddings)
                        retriever = vector_store.as_retriever(search_kwargs={"k": top_k}) # Retrieve top_k chunks
                        
                        # 3. Retrieve relevant chunks and construct the context
                        retrieved_docs = retriever.invoke(retrieval_query)
                        relevant_parts = "\n\n\n\n\n ".join([doc.page_content for doc in retrieved_docs])
                    else:
                        relevant_parts = full_text
                else:
                    # If specific source spans are provided, use them directly.
                    relevant_parts = get_relevant_text_sections(doc_path, source_spans)

                # Check that the length of the retrieved relevant parts remains below the maximum allowed number of characters for the context
                disclaimer = '... [context truncated due to length]'
                if len(relevant_parts) > max_context_length-len(disclaimer):
                    relevant_parts = relevant_parts[:max_context_length-len(disclaimer)]+disclaimer

                prompt_input = {
                    "json_field": key,
                    "relevant_parts": relevant_parts,
                    "generated_output": generated_output,
                    "ground_truth": ground_truth,
                    "formatting_instructions": formatting_instructions.get(key, ""),
                    "examples": examples.get(key, ""),
                    "output_format_instructions": parser.get_format_instructions()
                }

                try:
                    # The chain now directly returns a validated JudgeOutput object
                    evaluation_result = chain.invoke(prompt_input)
                    evaluation_scores = evaluation_result.scores
                    for criterion, score in evaluation_scores.items():
                        new_csv_line = {
                            'file': gt_file,
                            'field': key,
                            'criterion': criterion,
                            'score': score,
                            'explanation': evaluation_result.explanations.get(criterion, "")
                        }
                        # Write the new line in the csv file
                        writer.writerow(new_csv_line)

                except OutputParserException as e:
                    # Catch parsing errors specifically if the LLM fails to produce valid JSON
                    print(f"    - Error parsing LLM output for key '{key}': {e}")
                    continue
                except groq.RateLimitError as e:
                    # Check if the error is due to the daily token limit
                    if "tokens per day (TPD)" in str(e):
                        print(f"   - Daily token limit reached: {e}")
                        print("    - Stopping the script.")
                        # Close the file
                        f.close()
                        # Reorder the lines of the CSV file by using the first column entries ('file')
                        reordered_df = pd.read_csv(csv_output_filepath)
                        # Sort by multiple columns to ensure a fully deterministic order
                        reordered_df.sort_values(by=['file', 'field', 'criterion'], inplace=True)
                        reordered_df.to_csv(csv_output_filepath, index=False)
                        sys.exit(1)  # Exit the script
                    else:
                        # Handle other rate limit errors (e.g., tokens per minute)
                        print(f"   - A rate limit error occurred for key '{key}': {e}")
                        print("    - Waiting for 60 seconds before retrying.")
                        time.sleep(60)
                        continue
                except Exception as e: # Catch other errors, like from the API after retries
                    print(f"    - An unexpected error occurred for key '{key}': {e}")    
                    continue
                time.sleep(1.5) # Pace requests to the API to avoid rate-limiting.

    # Reorder the lines of the CSV file by using the first column entries ('file')
    reordered_df = pd.read_csv(csv_output_filepath)
    # Sort by multiple columns to ensure a fully deterministic order
    reordered_df.sort_values(by=['file', 'field', 'criterion'], inplace=True)
    reordered_df.to_csv(csv_output_filepath, index=False)
    print(f"\nLLM-as-a-judge evaluation saved to {csv_output_filepath}")

    end = time.time()
    print(f"Script completed in {end-start} seconds.")


### Function to compute the overall similarity score using the criteria scores assigned by the LLM judge
# Input arguments:
# - [str] judge_scores_csv_path: path to the CSV file containing scores and explanations assigned by the LLM judge
# - [str] output_csv_path: path to the CSV file where the overall scores should be saved
# - [List[float]] weight_list: list of weights to assign to each of the 5 criteria assigned by the LLM judge. The sum of its elements must be equal to 1.
def compute_overall_similarity_score(judge_scores_csv_path: str, output_csv_path: str, weight_list: List[float]=[0.35, 0.35, 0.1, 0.1, 0.1]) -> None:

    # Load the CSV file of scores assigned by the judge
    try:
        llm_judge_scores = pd.read_csv(judge_scores_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {judge_scores_csv_path}")
        return
    print('Processing the LLM judge scores in file '+judge_scores_csv_path+'...')

    # Define criteria and their corresponding weights
    criteria = ['correctness', 'completeness', 'clarity', 'conciseness', 'adherence']
    weights = dict(zip(criteria, weight_list))
    
    # Check for required columns in the input CSV
    required_cols = ['file', 'field', 'criterion', 'score']
    if not all(col in llm_judge_scores.columns for col in required_cols):
        missing = [col for col in required_cols if col not in llm_judge_scores.columns]
        print(f"Error: Input CSV is missing required columns: {missing}")
        return
    
    # Pivot the table to transform data from 'long' to 'wide' format. This creates columns for each criterion for each file-field pair.
    # If multiple scores exist for the same file-field-criterion combination, they will be averaged by default.
    print("Reshaping data from long to wide format ...")
    results = llm_judge_scores.pivot_table(
        index=['file', 'field'],
        columns='criterion',
        values='score',
        aggfunc='mean'
    ).reset_index()

    # Ensure all expected criteria columns exist, filling missing ones with NaN
    for criterion in criteria:
        if criterion not in results.columns:
            results[criterion] = np.nan
            print(f"Warning: Criterion '{criterion}' not found in input data. Its column will be empty.")

    # Convert score columns to numeric, coercing errors to NaN
    for criterion in criteria:
        results[criterion] = pd.to_numeric(results[criterion], errors='coerce')

    # Calculate the weighted average.
    # Pandas automatically handles NaNs: if any score in a row is NaN, the resulting overall_similarity_score for that row will also be NaN.
    # This fulfills the requirement to leave the overall score empty if any criterion score is missing.
    print("Calculating overall similarity scores...")
    weighted_sum = sum(results[c] * weights[c] for c in criteria if c in results.columns)
    total_weight = sum(weights.values())

    if total_weight > 0:
        results['overall_similarity_score'] = weighted_sum / total_weight
    else:
        results['overall_similarity_score'] = np.nan
    
    # Reorder columns for the final output file
    final_columns = ['file', 'field'] + criteria + ['overall_similarity_score']
    results = results[final_columns]

    # Save the results dataframe to the output path
    print(f"Saving results to {output_csv_path}...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results.to_csv(output_csv_path, index=False, encoding='utf-8')
    print("Processing complete.")


### Function to compute correlations between LLM judge and human scores
# Input arguments:
# - [str] llm_judge_scores_csv_path: path to the CSV file containing the scores assigned by the LLM judge
# - [str] human_scores_csv_path: path to the CSV file containing the scores assigned by the human
# - [str] output_csv_path: path to the CSV file containing correlation scores and p-values to be saved
def compute_llm_judge_human_correlations(llm_judge_scores_csv_path: str, human_scores_csv_path: str, output_csv_path: str) -> None:

    # Load the CSV files of human and LLM judge scores
    with open(llm_judge_scores_csv_path, 'r', encoding='utf-8') as f:
        llm_judge_scores = pd.read_csv(f)
    with open(human_scores_csv_path, 'r', encoding='utf-8') as f:
        human_scores = pd.read_csv(f)
    print("Calculating Spearman's Rank Correlation Coefficients between the human scores from file "+human_scores_csv_path+" and the LLM judge scores from file "+llm_judge_scores_csv_path+"...")

    # Identify the set of Bekanntmachungen that received both human and LLM annotations
    llm_evaluated_files = set(llm_judge_scores['file'])
    human_evaluated_files = set(human_scores['file'])
    common_files = llm_evaluated_files.intersection(human_evaluated_files)
    print("%d files with both human and LLM annotations found." % len(common_files))

    # Filter dataframes to only include common files
    human_scores_common = human_scores[human_scores['file'].isin(common_files)]
    llm_judge_scores_common = llm_judge_scores[llm_judge_scores['file'].isin(common_files)]

    # Merge the two dataframes to align scores
    merged_scores = pd.merge(
        human_scores_common,
        llm_judge_scores_common,
        on=['file', 'field', 'criterion'],
        suffixes=('_human', '_llm')
    )
    
    # Ensure scores are integers
    merged_scores['score_human'] = merged_scores['score_human'].astype(int)
    merged_scores['score_llm'] = merged_scores['score_llm'].astype(int)

    print("Computing correlations ...")

    # To preserve the original evaluation order, we can use Categorical types
    fields_to_evaluate = ['objective', 'inclusion_criteria', 'exclusion_criteria', 'deadline', 'max_funding', 'max_duration', 'procedure', 'contact', 'misc']
    criteria = ['correctness', 'completeness', 'clarity', 'conciseness', 'adherence']
    
    merged_scores = merged_scores[
        merged_scores['field'].isin(fields_to_evaluate) & 
        merged_scores['criterion'].isin(criteria)
    ]
    
    merged_scores['field'] = pd.Categorical(merged_scores['field'], categories=fields_to_evaluate, ordered=True)
    merged_scores['criterion'] = pd.Categorical(merged_scores['criterion'], categories=criteria, ordered=True)

    # Group by field and criterion and apply the correlation function
    result_frame = merged_scores.groupby(['field', 'criterion'], sort=False, observed=False).apply(spearman_corr_custom).reset_index()

    # Save the resulting pandas frame to CSV
    result_frame.to_csv(output_csv_path, index=False)
    print("Correlations saved to "+output_csv_path+".")


# ### Function to compute correlations between LLM judge and human scores
# def compute_llm_judge_human_correlations(llm_judge_scores_csv_path: str, human_scores_csv_path: str, output_csv_path: str) -> None:

#     # Load the CSV files of human and LLM judge scores
#     with open(llm_judge_scores_csv_path, 'r', encoding='utf-8') as f:
#         llm_judge_scores = pd.read_csv(f)
#     with open(human_scores_csv_path, 'r', encoding='utf-8') as f:
#         human_scores = pd.read_csv(f)
#     print("Calculating Spearman's Rank Correlation Coefficients between the human scores from file "+human_scores_csv_path+" and the LLM judge scores from file "+llm_judge_scores_csv_path+"...")

#     # Identify the set of Bekanntmachungen that received both human and LLM annotations
#     llm_evaluated_files = set(llm_judge_scores['file'])
#     human_evaluated_files = set(human_scores['file'])
#     common_files = llm_evaluated_files.intersection(human_evaluated_files)
#     print("%d files with both human and LLM annotations found." % len(common_files))

#     # Create a dictionary to store lists of scores, with keys 'human' and 'llm'. 
#     # The two values are dictionaries with keys being fields to evaluate (e.g. 'objective', 'inclusion_criteria', etc.).
#     # Each field value is a dictionary with keys being the 5 criteria.
#     # Each criteria value is a list of size number of Bekanntmachungen.
#     scores = {'human': {}, 'llm': {}}
#     fields_to_evaluate = ['objective', 'inclusion_criteria', 'exclusion_criteria', 'deadline', 'max_funding', 'max_duration', 'procedure', 'contact', 'misc']
#     criteria = ['correctness', 'completeness', 'clarity', 'conciseness', 'adherence']
    
#     for field in fields_to_evaluate:
#         scores['human'][field] = {}
#         scores['llm'][field] = {}
#         for criterion in criteria:
#             human_scores_list = []
#             llm_scores_list = []
#             for file in common_files:
#                 human_scores_list.append(int(human_scores[human_scores['file']==file][human_scores['field']==field][human_scores['criterion']==criterion]['score'].iloc[0]))
#                 llm_scores_list.append(int(llm_judge_scores[llm_judge_scores['file']==file][llm_judge_scores['field']==field][llm_judge_scores['criterion']==criterion]['score'].iloc[0]))
#             scores['human'][field][criterion] = human_scores_list
#             scores['llm'][field][criterion] = llm_scores_list
                
#     # Loop on the combinations of field+criteria to compute the Spearmann's rank correlation coefficient and save them in a panda data frame
#     records = []

#     print("Computing correlations ...")
#     for field in fields_to_evaluate:
#         for criterion in criteria:
#             human_scores_list = scores['human'][field][criterion]
#             llm_scores_list = scores['llm'][field][criterion]
#             correlation, p_value = spearmanr(human_scores_list, llm_scores_list)
#             records.append({'field': field, 'criterion': criterion, 'spearman_correlation': correlation, 'p_value': p_value})

#     # Save the resulting panda frame to CSV
#     result_frame = pd.DataFrame(records)
#     result_frame.to_csv(output_csv_path, index=False)
#     print("Correlations saved to "+output_csv_path+".")


### Main function
if __name__ == '__main__':

    # Load the YAML config file and main logic
    try:
        config = load_config_from_yaml(r'./config.yaml')
    except:
        print("ERROR: YAML config file './config.yaml' not found!")
        sys.exit(1)

    run_steps = config.get('run_steps', {})

    # Compute BLEU, ROUGE and BertScore metrics
    if run_steps.get('compute_traditional_metrics'):
        metrics_parameters = config.get('compute_traditional_metrics', {})
        compute_metrics(
            gt_path=metrics_parameters.get('gt_path'), 
            gen_path=metrics_parameters.get('gen_path'), 
            output_filepath=metrics_parameters.get('output_filepath')
            )
        
    # Plot boxplots and violin plots of metrics per json field
    if run_steps.get('plot_traditional_metrics'):
        plot_parameters = config.get('plot_traditional_metrics', {})
        plot_metrics(
            csv_filepath=plot_parameters.get('csv_filepath'), 
            output_folder_path=plot_parameters.get('output_folder_path')
            )

    # Run LLM-as-a-judge using Groq
    if run_steps.get('run_llm_as_judge'):
        llm_judge_parameters = config.get('run_llm_as_judge', {})
        llm_judge = llm_judge_parameters.get('model')
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found among the environment variables defined in .env")
        print("Evaluating with " + llm_judge + " on Groq...")
        groq_llm = ChatGroq(
            model=llm_judge,
            api_key=groq_api_key,
            temperature=llm_judge_parameters.get('temperature'), 
        )
        #llm_score_csv_path = r'.\evaluation\llm_as_judge\gemini-2.5-flash_judged_by_'+llm_judge+'.csv'
        llm_as_a_judge_evaluation(
            gt_path=llm_judge_parameters.get('gt_path'),
            gen_path=llm_judge_parameters.get('gen_path'),
            text_path=llm_judge_parameters.get('text_path'),
            csv_output_filepath=llm_judge_parameters.get('csv_output_filepath'),
            llm=groq_llm
        )

    # Compute the overall LLM judge scores for each file and save them in a csv file
    # overall_scores_csv_path = r'.\evaluation\llm_as_judge\gemini-2.5-flash_judged_by_'+llm_judge+'_overall_scores.csv'
    # compute_overall_similarity_score(judge_scores_csv_path=llm_score_csv_path, output_csv_path=overall_scores_csv_path, weight_list=[0.35, 0.35, 0.1, 0.1, 0.1])
    if run_steps.get('compute_overall_llm_judge_scores'):
        overall_judge_scores_parameters = config.get('compute_overall_llm_judge_scores', {})
        weight_list = [
            overall_judge_scores_parameters.get('scoring_weights')['correctness'],
            overall_judge_scores_parameters.get('scoring_weights')['completness'],
            overall_judge_scores_parameters.get('scoring_weights')['clarity'],
            overall_judge_scores_parameters.get('scoring_weights')['conciseness'],
            overall_judge_scores_parameters.get('scoring_weights')['adherence']
        ]
        compute_overall_similarity_score(
            judge_scores_csv_path=overall_judge_scores_parameters.get('judge_scores_csv_path'),
            output_csv_path=overall_judge_scores_parameters.get('output_csv_path'),
            weight_list=weight_list
        )

    # Compute the Spearman's Rank Correlation Coefficient between human and LLM judge scores
    if run_steps.get('compute_spearman_correlations'):
        overall_correlation_parameters = config.get('compute_spearman_correlations', {})
        compute_llm_judge_human_correlations(
            llm_judge_scores_csv_path=overall_correlation_parameters.get('llm_judge_scores_csv_path'),
            human_scores_csv_path=overall_correlation_parameters.get('human_scores_csv_path'),
            output_csv_path=overall_correlation_parameters.get('output_csv_path')
        )