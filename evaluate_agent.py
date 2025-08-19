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
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
#from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tenacity
from langchain_groq import ChatGroq
import time
import requests
import groq
import csv
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
        max_context_length: int=16000,
        chunk_size: int=1500,
        chunk_overlap: int=150,
        top_k: int=5) -> None:

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
    results = []

    # Define the output parser that also defines formatting instructions
    # Pydantic model for the LLM-as-a-judge output
    class JudgeOutput(BaseModel):
        scores: Dict[str, int] = Field(description="A dictionary where keys are the evaluation criteria (correctness, completeness, clarity, conciseness, adherence) and values are the scores from 1 to 5.")
        explanations: Dict[str, str] = Field(description="A dictionary where keys are the evaluation criteria and values are the textual explanations for the scores.")
    
    # Use a self-correcting parser that can fix malformed JSON from the LLM
    parser = PydanticOutputParser(pydantic_object=JudgeOutput)
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
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

    # NOTE: old format instructions:
    # "Return both your scores and explanations in a json format with two keys called 'scores' and 'explanations'.\n\
    # The value for 'scores' should be a dictionary where each of the five criteria is a key, with the given score as its associated value.\n\
    # The value for 'explanations' should be a dictionary where each of the five criteria is a key, with your explanations in text format as its associated value.\n\
    # For example: {'scores': {'correctness': 4, 'completeness': 3, 'clarity': 5, 'conciseness': 2, 'adherence': 1}, \
    # 'explanations': {'correctness': '...', 'completeness': '...', 'clarity': '...', 'conciseness': '...', 'adherence': '...'}}"

    # Specific formatting instructions for each json field
    formatting_instructions = {
        "objective": "A high-level summary (e.g. 2 to 5 sentences) of the main objective(s) and topic(s) of the call described in the grant document should be provided.\n\
            Although this field can be open-ended, a particular interest is placed on whether the investigation of some topics or themes is encouraged within the call, \
            on what is expected to be developped in the project, for which target group(s), and how.",
        "inclusion_criteria": "Any inclusion criteria that applies to the applicants of the call described in the grant document should be listed.\n\
            Descriptions of the inclusion criteria are expected to be relatively brief.\n\
            Information of particular interest includes what type of institutions are allowed to apply, and if any important requirement applies to the applicants.\n\
            If no inclusion criteria can be found, 'None' should be returned.",
        "exclusion_criteria": "Any exclusion criteria that may apply to the applicants of the call described in the grant document should be listed.\n\
            Information of interest for instance includes whether some particular types of institutions are not allowed to apply to the call, \
            or if the participants must not come from specific parts of the world.\n\
            If no exclusion criteria can be found, 'None' should be returned.",
        "deadline": "The deadline for the first stage of the application should be formatted as DD.MM.YYYY.\n\
            As the original grant document may contain several non-deadline dates, specific care should be taken that the correct deadline is identified.\n\
            If several deadlines are present in the document, they should follow the aforementioned format, and be separared by semicolons.\n\
            If no deadline is specified, 'None' should be returned.",
        "max_funding": "The maximum funding amount should be formatted using comma as thousand separators and dot as decimal point, with the associated currency (e.g. euros, dollars).\n\
            If the information is available in the grant document, it should be specified if the amount applies to the whole consortium or to a single partner.\n\
            If no maximum funding amount can be found, 'Not specified' or 'Not specified.' should be returned.",
        "max_duration": "The maximum duration for the project should be returned in months, and formatted as a number followed by 'months'.\n\
            If no maximum duration can be found, 'Not specified' or 'Not specified.' should be returned.",
        "procedure": "Concrete stages of the procedure to submit an application to the grant call should be listed.\n\
            The number of stages (usually 1, 2 or 3) is in particular important.\n\
            A detailed description of each stage is not necessarily expected, but additional practical details regarding the first stage such as language and maximum number of pages allowed for the application, \
            whether the document must be submitted electronically or in written form should be provided.\n\
            If no information about the procedure can be found, 'None' should be returned.",
        "contact": "The contact information should include whenever possible name, e-mail and phone number, in order and separated by commas.\n\
            If several contacts are specified, all contacts should follow the aforementioned format and be separated by semicolons.\n\
            If several contacts share the same contact information (e.g. e-mail, phone number), the information should be repeated for each contact.\n\
            If no contact information can be found, 'None' should be returned.",
        "misc": "Any additional information that can be useful to the applicant can be listed here.\n\
            This field can be open-ended, but information of specific interest includes whether the de-minimis rule applies, what is the Förderquote for different partners, \
            whether specific institutions are expressly invited to apply, whether geographical restrictions apply to the results of the project, \
            or whether an expected starting time for the accpeted projects is provided.\n\
            If no specific additional information can be found, 'None' should be returned."
        }
    
    # Examples provided for each json field
    examples = {
        "objective": "*Example:* 1.1 Förderziel\n\
            Künstliche Intelligenz (KI) durchdringt Wirtschaft, Wissenschaft und Gesellschaft. Technologische Weiterentwicklungen ermöglichen neue Anwendungsmöglichkeiten. \
            Die deutsche KI-Forschungslandschaft zählt zu den besten weltweit, was sich beispielweise an der regelmäßigen Platzierung unter den fünf publikationsstärksten und \
            meistzitierten Ländern zeigt. Dies gilt nicht nur für KI allgemein, sondern auch für zentrale Teilbereiche wie Sprachtechnologien oder Robotik. Deutschland besitzt \
            damit gute Voraussetzungen, um an der Gestaltung zukünftiger KI-Anwendungen mitzuwirken und so die technologische Souveränität in dieser wichtigen Schlüsseltechnologie \
            zu sichern und auszubauen. Zugleich stellen sich weiterhin viele – teilweise grundlegende – Forschungsfragen. Auch bei der Übersetzung der Forschung in neue Anwendungen \
            und wirtschaftlichen Mehrwert muss Deutschland im internationalen Wettbewerb zielgerichteter und wirksamer agieren.\n\
            Für den Erfolg von Forschung, Transfer und wirtschaftlicher Anwendung sind vor allem gut ausgebildete KI-Fachkräfte in Wissenschaft und Anwendung erforderlich. \
            Diese Fachkräfte in Deutschland auszubilden, zu halten und weitere aus dem Ausland hinzuzugewinnen, sind angesichts eines schärfer werdenden internationalen \
            Fachkräftewettbewerbs zentrale Herausforderungen für die Bundesregierung. Insbesondere die Beteiligung von Frauen in der deutschen KI-Forschung und in entsprechenden \
            akademischen Führungspositionen entspricht nicht dem Anteil herausragend qualifizierter Frauen in der Bevölkerung. Dadurch wird ein großes Potenzial nicht genutzt, zum \
            Nachteil des Forschungsstandorts. Mit der Förderung „Exper Team 4 KI“ zielt das Bundesministerium für Bildung und Forschung (BMBF) darauf ab, Nachwuchswissenschaftlerinnen \
            im Bereich Künstliche Intelligenz beim Aufbau einer eigenen Forschungsgruppe und der eigenständigen Arbeit an innovativen, fachübergreifenden Forschungsideen zu unterstützen.\n\
            Das BMBF hat hierzu bereits in der Vergangenheit wichtige Initiativen angestoßen: Mit der Richtliniezur Förderung von KI-Nachwuchswissenschaftlerinnen vom 4. Juni 2019 \
            (BA nz AT 19.06.2019 B 5) wurde in einem ersten Schritt die stärkere Berücksichtigung, die umfassendere Beteiligung und der stärkere Einfluss von Frauen in der KI-Forschung \
            angestrebt. Darauf aufbauend wurden mit der Richtliniezur Förderung von Ideennachwuchs: Forschungsvorhaben von KI-Nachwuchsgruppen vom 15. April 2021 (BA nz AT 07.05.2021 B 6) \
            interdisziplinäre Nachwuchsgruppen rund um das Themenfeld Künstliche Intelligenz gefördert.\n\
            Zugleich bleibt die Nachfrage nach gut qualifizierten KI-Fachkräften auch in der Wissenschaft hoch. Zudem war auch 2022 nur an etwa jeder dritten KI-Publikation aus Deutschland \
            mindestens eine Wissenschaftlerin beteiligt. Mit der vorliegenden Richtlinie zur KI-Nachwuchsförderung setzt das BMBF daher einen weiteren Impuls, um Nachwuchsforscherinnen im \
            Bereich KI zu fördern, die Forschung zum Thema KI in Deutschland weiter voranzubringen und die verantwortungsvolle Beteiligung von herausragend qualifiziertem Nachwuchs im \
            Wissenschaftssystem sowie in der forschungsnahen Wirtschaft durch Transfer und Ausgründungen weiter zu erhöhen. Dabei können auch herausragende Forschungsvorhaben, die auf die \
            Ausgründung eines „Start-up“-Unternehmens abzielen, unterstützt werden.\n\
            2.1 Forschungsgegenstand\n\
            Die Nachwuchsgruppen sollten Arbeiten an einem oder mehreren der nachfolgenden Themengebiete durchführen:\n\
            - Grundlagen der KI: Zuverlässigkeit, Wissensrepräsentation, Umgang mit Unsicherheite\n\
            - Maschinelles Lernen: neue Lernmethoden, Robustheit, Validierungsverfahren\n\
            - Ressourceneffiziente KI-Systeme: daten- und/oder energiesparsame KI, Optimierung der Performance von KI in Training und Inferenz\n\
            - Hybride KI: Integration von maschinellen Lernverfahren und modellbasierter KI\n\
            - KI-basierte Datenanalyse und Wissensextraktion: Sprach-, Text- und Bildverstehen, multimodales Lernen, Knowledge Refinement\n\
            Vorhaben mit Fokus auf andere Themen sind in begründeten Ausnahmen möglich. Es gelten die nachfolgend genannten Einschränkungen.\n\
            Um Überschneidungen zu anderen Förderbereichen zu vermeiden und die Breite der Forschungsfelder zu erhöhen, werden im Rahmen dieser Bekanntmachung keine Projekte gefördert, \
            die den Einsatz von KI in der Medizin, für das Personalwesen, im Marketing oder der Kundenbetreuung, für die IT-Sicherheit, „Predictive Maintenance“, im Bereich ziviler Sicherheit \
            oder von robotischen Systemen für die Pflege zum Ziel haben. Weiterhin muss in anwendungsgetriebenen oder interdisziplinären Vorhaben ein Mehrwert für die KI-Forschung entstehen.\n\
            Der Praxisbezug der anwendungsorientierten Projekte und die Verwertbarkeit der Ergebnisse sind durch eine angemessene Einbindung von Anwendenden aus der gewerblichen Wirtschaft \
            (als assoziierte Projektpartner) sicherzustellen. Grundsätzlich ist zu beachten: Die Neuentwicklung und Adaption von ausschließlich innerbetrieblich genutzten Basiskomponenten \
            sind von der Förderung ausgeschlossen.\n\
        *Expected answer:* The call funds the establishment of junior research groups led by women in the field of artificial intelligence. The following subject areas are more specifically targetted by this call: \
            fundamentals of AI, machine learning, resource-efficient AI systems, hybrid AI, AI-based data analysis and knowledge extraction.",
        "inclusion_criteria": "*Example:* 3 Zuwendungsempfänger\n\
            Antragsberechtigt sind Vereine, Stiftungen, Bildungsinstitutionen, einschlägige wissenschaftliche Einrichtungen aus der Transfer-, Kommunal- oder Bildungsforschung sowie vergleichbare Institutionen \
            (einschließlich Hochschulen und Forschungseinrichtungen) in Einzel- oder Verbundvorhaben, die im kommunalen Bildungsmanagement beziehungsweise den Schwerpunktthemen der Fachstellen ausgewiesen sind. \
            Zum Zeitpunkt der Auszahlung einer gewährten Zuwendung wird das Vorhandensein einer Einrichtung, die der nichtwirtschaftlichen Tätigkeit des Zuwendungsempfängers dient (Hochschule, Forschungseinrichtung, \
            Verein, Stiftung, Bildungsinstitution), in Deutschland verlangt.\n\
            Forschungseinrichtungen, die von Bund und/oder Ländern grundfinanziert werden, können neben ihrer institutionellen Förderung nur unter bestimmten Voraussetzungen eine Projektförderung für ihre zusätzlichen \
            projektbedingten Ausgaben beziehungsweise Kosten bewilligt bekommen.\n\n\
            *Expected answer:* Eligible to apply are associations, fondations, educational institutions, scientific institutions from transfer, municipal or educational research (such as universities and research institutions, \
            based in Germany.",
        "exclusion_criteria": "*Example:* 2.1 Forschungsgegenstand\n\
            Die Nachwuchsgruppen sollten Arbeiten an einem oder mehreren der nachfolgenden Themengebiete durchführen:\n\
            - Grundlagen der KI: Zuverlässigkeit, Wissensrepräsentation, Umgang mit Unsicherheite\n\
            - Maschinelles Lernen: neue Lernmethoden, Robustheit, Validierungsverfahren\n\
            - Ressourceneffiziente KI-Systeme: daten- und/oder energiesparsame KI, Optimierung der Performance von KI in Training und Inferenz\n\
            - Hybride KI: Integration von maschinellen Lernverfahren und modellbasierter KI\n\
            - KI-basierte Datenanalyse und Wissensextraktion: Sprach-, Text- und Bildverstehen, multimodales Lernen, Knowledge Refinement\n\
            Vorhaben mit Fokus auf andere Themen sind in begründeten Ausnahmen möglich. Es gelten die nachfolgend genannten Einschränkungen.\n\
            Um Überschneidungen zu anderen Förderbereichen zu vermeiden und die Breite der Forschungsfelder zu erhöhen, werden im Rahmen dieser Bekanntmachung keine Projekte gefördert, \
            die den Einsatz von KI in der Medizin, für das Personalwesen, im Marketing oder der Kundenbetreuung, für die IT-Sicherheit, „Predictive Maintenance“, im Bereich ziviler \
            Sicherheit oder von robotischen Systemen für die Pflege zum Ziel haben. Weiterhin muss in anwendungsgetriebenen oder interdisziplinären Vorhaben ein Mehrwert für die KI-Forschung entstehen.\n\
            Der Praxisbezug der anwendungsorientierten Projekte und die Verwertbarkeit der Ergebnisse sind durch eine angemessene Einbindung von Anwendenden aus der gewerblichen Wirtschaft (als assoziierte Projektpartner) \
            sicherzustellen. Grundsätzlich ist zu beachten: Die Neuentwicklung und Adaption von ausschließlich innerbetrieblich genutzten Basiskomponenten sind von der Förderung ausgeschlossen.\n\n\
            *Expected answer:* This call does not fund projects that aim to using AI in medicine, for human resources, in marketing or customer service, for IT security, 'Predictive Maintenance', in the field of civil security, or robotic systems for care.",
        "deadline": "*Example:* Richtlinie im Rahmen der Initiative zur Förderung von Projekten zum Thema „Transformation fördern“, Bundesanzeiger vom 31. Januar 2024.\n\
            Im ersten Verfahrensschritt sind Projektskizzen bis spätestens 25. Oktober 2024 schriftlich und/oder elektronisch über die Online-Einreichungsplattform beim ABC-Projektträger einzureichen.\n\
            Berlin, 17.02.2024\n\n\
            *Expected answer:* 25.10.2024",
        "max_funding": "*Example:* Bemessungsgrundlage für Zuwendungen an Hochschulen, Forschungs- und Wissenschaftseinrichtungen sowie vergleichbare Einrichtungen, die nicht dem wirtschaftlichen Tätigkeitsbereich \
            zuzuordnen sind, sind die oben genannten förderfähigen projektbezogenen Ausgaben (Personal- und Dienstreisen), die unter Berücksichtigung beihilferechtlicher Vorgaben auch anteilig bis zu 100 Prozent \
            und maximal bis zu 900.000 Euro pro Vorhaben für den Bewilligungszeitraum gefördert werden können.\n\n\
            *Expected answer:* 900,000 euros per project",
        "max_duration": "*Example:* Die Zuwendungen werden als Projektförderung für einen Förderzeitraum von bis zu drei Jahren als nicht rückzahlbare Zuschüsse gewährt. \
            Der Förderbeginn für Projektanträge ist für den 31. März 2017 vorgesehen. Die Bewilligungsbehörde (vgl. Abschnitt 7.1) behält sich einen abweichenden Beginn vor.\n\n\
            *Expected answer:* 36 months",
        "procedure": "*Example:* 7.2 Zweistufiges Antragsverfahren\n\
            Das Antragsverfahren ist zweistufig angelegt.\n\
            7.2.1 Vorlage und Auswahl von Projektskizzen\n\
            In der ersten Verfahrensstufe sind dem beauftragten Projektträgerbis spätestens 31.03.2023 zunächst Projektskizzen in schriftlicher und/oder elektronischer Form vorzulegen.\n\
            Bei Verbundprojekten sind die Projektskizzen in Abstimmung mit dem vorgesehenen Verbundkoordinator vorzulegen. Die Vorlagefrist gilt nicht als Ausschlussfrist. Projektskizzen, die nach dem oben angegebenen Zeitpunkt eingehen, können aber möglicherweise nicht mehr berücksichtigt werden.\n\
            Die Projektskizze soll enthalten:\n\
            - kurze Darstellung der Organisation: Struktur und Ziele, einschlägige Erfahrungen und Kompetenzen\n\
            - Aufzählung einschlägiger Publikationen (nur bei Fördergegenstand Fachstellen)\n\
            - Definition und Begründung der regionalen Ausrichtung und Standortwahl (nur bei Fördergegenstand REAB)\n\
            - Beschreibung des Gesamtziels des Vorhabens und Bezug zu den förderpolitischen Zielen des Programms\n\
            - Skizzierung des Umsetzungskonzepts und zentraler Inhalte\n\
            - Arbeitsteilung/Zusammenarbeit mit Dritten (insbesondere Beschreibung der Zusammenarbeit im Netzwerk der REAB und Fachstellen sowie mit der kommunalen Landschaft)\n\
            - geschätzte Gesamtausgaben\n\
            Die Projektskizze soll maximal 10 Seiten (DIN A 4, 1, 5-zeilig, Arial in Schriftgröße 11) umfassen. Anlagen sind nicht beizufügen.\n\
            Die eingegangenen Projektskizzen werden nach den folgenden Kriterien bewertet:\n\
            - Beitrag des geplanten Vorhabens zur Erreichung der förderpolitischen Ziele\n\
            - Fachliche Qualität und Plausibilität der Umsetzungsskizze\n\
            - Plausibilität der Überlegungen zu Arbeitsteilung und Zusammenarbeit\n\
            - Plausibilität der geschätzten Gesamtausgaben\n\
            Entsprechend der oben angegebenen Kriterien und Bewertung werden die für eine Förderung geeigneten Projektskizzen ausgewählt. Das Auswahlergebnis wird den Interessenten schriftlich mitgeteilt.\n\
            Die im Rahmen dieser Verfahrensstufe eingereichte Projektskizze und eventuell weitere vorgelegte Unterlagen werden nicht zurückgesendet.\n\
            7.2.2 Vorlage förmlicher Förderanträge und Entscheidungsverfahren\n\
            In der zweiten Verfahrensstufe werden die Verfasser der positiv bewerteten Projektskizzen aufgefordert, einen förmlichen Förderantrag vorzulegen.\n\
            Bei Verbundprojekten sind die Förderanträge in Abstimmung mit dem vorgesehenen Verbundkoordinator vorzulegen.\n\
            Die Anträge sind spätestens bis zum 31.08.2023 in schriftlicher und/oder elektronischer Form beim beauftragten Projektträger einzureichen.\n\
            Anträge, die nach dem oben angegebenen Zeitpunkt eingehen, können möglicherweise nicht mehr berücksichtigt werden.\n\
            Mit dem Förderantrag ist eine ausführliche Vorhabenbeschreibung einzureichen (max. 12 Seiten DIN A 4, 1, 5-zeilig, Arial in Schriftgröße 11). Diese soll enthalten:\n\
            - Darlegung einschlägiger Kompetenzen für die Projektumsetzung\n\
            - Darstellung des Eigeninteresses des Antragstellers an dem Vorhaben\n\
            - Einordnung des Gesamtziels des Vorhabens in die förderpolitischen Ziele des Programms\n\
            - Konzept für die Entwicklung (kommunaler) Beratungs- und Unterstützungsprozesse im Rahmen des Netzwerks von REAB und Fachstellen für kommunales Bildungsmanagement\n\
            - Beschreibung der Einbindung der Agentur in regionale beziehungsweise ländereigene Abstimmungsstrukturen im Bildungssystem, Kooperationen mit regionalen und kommunalen Landesverbänden sowie Akteuren der Zivilgesellschaft und Bildungslandschaft (nur Fördergegenstand REAB)\n\
            - Beschreibung der Einbindung der Fachstelle in den Fachdiskurs zum Bildungsmanagement (nur Fördergegenstand Fachstellen)\n\
            - Darstellung der wissenschaftlichen und technischen Arbeitsziele\n\
            - Verwertungsplan und Nachhaltigkeitsperspektiven\n\
            - Darstellung der Notwendigkeit der Zuwendung\n\
            Als Anlagen sind beizufügen:\n\
            - Arbeits- und Zeitplan,\n\
            - ressourcenbezogener Arbeitsplan unter Ausweisung von Mensch-Monaten für das im Projekt tätige Personal,\n\
            - Tätigkeitsprofile für die geplanten Personalstellen.\n\
            Die eingegangenen Anträge werden nach den folgenden Kriterien bewertet und geprüft:\n\
            - Inhaltliche und fachliche Passgenauigkeit der Kompetenzen, die eingebracht werden sollen\n\
            - Zielgenauigkeit des Verständnisses für die förderpolitischen Ziele\n\
            - Qualität des Beratungs- und Unterstützungskonzepts mit Bezug auf die Förderziele der Module der Förderrichtlinie „Bildungskommunen“: datenbasiertes kommunales Bildungsmanagement, analog-digital vernetzte Bildungslandschaft und thematische Schwerpunkte\n\
            - Plausibilität der Verortung der REAB/Fachstelle in Kooperations- und Prozessstrukturen des Fachnetzwerks\n\
            - Plausibilität der Arbeits- und Zeitplanung\n\
            - Nachhaltige Anschlussperspektive im Rahmen der Verwertungsplanung\n\
            - Nachvollziehbare Planung der Gesamtausgaben\n\
            - Einschätzung der Verwertungs-/Anwendungsmöglichkeiten\n\n\
            *Expected answer:* The procedure is two-stage. In the first one, project outlines of maximum 10 pages must be submitted either in written and/or electronic forms to the deignated project management organisation before the specified deadline.\
            After evaluation, accepted applications will proceed to a second stage involving the submission of a formal funding application.",
        "contact": "*Example:* Mit der Durchführung der Fördermaßnahme hat das BMBF derzeit folgenden Projektträger beauftragt: ABC Projektträger - Abteilung Forschung - Berliner-Straße 56 6701 München \
            Telefon: +49 228 3821-1210 Telefax: +49 228 3821-1257 Ansprechpartner sind: Dr. Alice Schmidt Telefon: +49 123 4567-8910 E-Mail: alice.schmidt@abc.de Herr Robert Müller Telefon: +49 987 6543-2100 \
            E-Mail: robert.mueller@abc.de Internet www.some-website.de\n\n\
            *Expected answer:* Dr. Alice Schmidt, alice.schmidt@abc.de, +49 123 4567-8910; Robert Müller, robert.mueller@abc.de, +49 987 6543-2100",
        "misc": "*Example 1:* 1.2 Zuwendungszweck\n\
            Zuwendungszweck ist die Erforschung von KI-Fragestellungen zu neuartigen und innovativen Themen durch von Frauen geleitete KI-Nachwuchsgruppen. Die Förderung soll \
            Wissenschaftlerinnen den Aufbau einer eigenständigen Arbeitsgruppe ermöglichen, ihr wissenschaftliches Profil stärken und ihre Sichtbarkeit in der Community erhöhen. \
            Bei anwendungsorientierten Forschungsthemen soll zudem der Transfer von Ideen in die Wirtschaft erleichtert werden.\n\
            Die Fördermaßnahme dient der Umsetzung der KI-Strategie der Bundesregierung und ihrer Fortschreibung sowie der Zukunftsstrategie für Forschung und Innovation.\
            Das BMBF erwartet, dass die Maßnahme den Anteil qualifizierter Frauen in Führungspositionen in der deutschen KI-Forschung erhöht und den Einfluss von Forscherinnen \
            in diesem Bereich nachhaltig stärkt.\n\
            Die Ergebnisse des geförderten Projekts dürfen nur in der Bundesrepublik Deutschland bzw. im Europäischen Wirtschaftsraum (EWR) und der Schweiz genutzt werden.\n\n\
            *Expected answer 1:* The results of the project can only be used in Germany, Switzerland or the European Economic Area (EEA).\n\n\
            *Example 2:* 1.3 Rechtsgrundlagen\n\
            Der Bund gewährt die Zuwendungen nach Maßgabe dieser Förderrichtlinie, der §§ 23 und 44 der Bundeshaushaltsordnung (BHO) und den dazu erlassenen Verwaltungsvorschriften \
            sowie der „Richtlinien für Zuwendungsanträge auf Ausgabenbasis (AZA/AZAP/AZV)“ und/oder der „Richtlinien für Zuwendungsanträge auf Kostenbasis von Unternehmen der \
            gewerblichen Wirtschaft (AZK)“ BMBF. Ein Anspruch auf Gewährung der Zuwendung besteht nicht.\n\
            Vielmehr entscheidet die Bewilligungsbehörde aufgrund ihres pflichtgemäßen Ermessens im Rahmen der verfügbaren Haushaltsmittel.\n\
            Nach dieser Förderrichtlinie werden staatliche Beihilfen auf der Grundlage von Artikel 25 Absatz 1 und 2 Buchstabe b und c und Artikel 28 Absatz 1 der Allgemeinen \
            Gruppenfreistellungsverordnung (AGVO) der EU-Kommission gewährt.4 Die Förderung erfolgt unter Beachtung der in Kapitel I AGVO festgelegten Gemeinsamen Bestimmungen, \
            insbesondere unter Berücksichtigung der in Artikel 2 der Verordnung aufgeführten Begriffsbestimmungen (vergleiche hierzu die Anlage zu beihilferechtlichen Vorgaben für \
            die Förderrichtlinie).\n\
            Nach dieser Förderrichtlinie werden staatliche Beihilfen im Sinne der De-minimis-Beihilfen Verordnung der Europäischen Kommission (EU-Kommission) gewährt.\n\
            5 Art und Umfang, Höhe der Zuwendung\n\
            Die Zuwendungen werden im Wege der Projektförderung gewährt.\n\
            Bemessungsgrundlage für Zuwendungen an Unternehmen der gewerblichen Wirtschaft und für Vorhaben von Forschungseinrichtungen, die in den Bereich der wirtschaftlichen \
            Tätigkeiten 5 fallen, sind die zuwendungsfähigen projektbezogenen Kosten. In der Regel können diese - je nach Anwendungsnähe des Vorhabens - unter Berücksichtigung der \
            beihilferechtlichen Vorgaben (siehe Anlage) bis zu 50 % anteilfinanziert werden. Nach BMBF-Grundsätzen wird eine angemessene Eigenbeteiligung - grundsätzlich mindestens \
            50 % der entstehenden zuwendungsfähigen Kosten - vorausgesetzt.\n\
            Bemessungsgrundlage für Zuwendungen an Hochschulen, Forschungs- und Wissenschaftseinrichtungen und vergleichbare Institutionen, die nicht in den Bereich der \
            wirtschaftlichen Tätigkeiten fallen, sind die zuwendungsfähigen projektbezogenen Ausgaben (bei Helmholtz-Zentren und der Fraunhofer-Gesellschaft die zuwendungsfähigen \
            projektbezogenen Kosten), die unter Berücksichtigung der beihilferechtlichen Vorgaben individuell bis zu 100 % gefördert werden können.\n\
            Bei nichtwirtschaftlichen Forschungsvorhaben an Hochschulen und Universitätskliniken wird zusätzlich zu den zuwendungsfähigen Ausgaben eine Projektpauschale in Höhe von 20 % gewährt.\
            Für die Festlegung der jeweiligen zuwendungsfähigen Kosten und für die Bemessung der jeweiligen Förderquote muss die AGVO berücksichtigt werden (siehe Anlage).\n\
            *Expected answer 2:* The de-minimis rule applies to this call. The Förderquote is 100% for Universities and Research Institutions, and 50% for commercial enterprises. \
            The average Förderquote of the whole consortium should be at most 50%.\n\n\
            *Example 3:* Der Projektstart soll ab dem 1. Juni 2019 erfolgen. Es wird erwartet, dass die Qualifikationsstelle innerhalb der ersten drei Monate nach Laufzeitbeginn besetzt wird. \
            Eingereichte Förderanträge und weitere vorgelegte Unterlagen werden nicht zurückgesendet.\n\
            *Expected answer 3:* The project is expected to start on 01.06.2019.\n\n\
            *Example 4:* 1.1 Zuwendungszweck\n\
            Das Ziel dieser gemeinsamen deutsch-südafrikanischen Bekanntmachung ist die Intensivierung der Zusammenarbeit zwischen Südafrika und Deutschland in den Bereichen Wissenschaft, Forschung und Technologie. \
            Es sollen sowohl bestehende Kooperationen ausgebaut, als auch neue Projektkooperationen initiiert werden. Die geförderten Vorhaben sollen auch der Vorbereitung von umfangreicheren \
            Antragstellungen bei Förderorganisationen wie z. B. Bundesministerium für Bildung und Forschung (BMBF), Deutscher Forschungsgemeinschaft oder Europäischer Union (EU) dienen.\n\
            Bei den gemeinsamen Projekten wird besonderer Wert auf die wissenschaftliche Exzellenz der südafrikanischen und der deutschen Partner sowie die Einbindung von wissenschaftlichem \
            Nachwuchs aus beiden Ländern gelegt. Besonders begrüßt wird die Beteiligung von Unternehmen, insbesondere von innovativen kleinen und mittleren Unternehmen (KMU) aus Deutschland.\n\
            Diese Bekanntmachung basiert auf dem Abkommen zur Zusammenarbeit im Bereich von Wissenschaft, Forschung und Technologie zwischen Deutschland und Südafrika vom 12. Juni 1996.\n\
            *Expected answer 4:* The participation of companies, especially small and medium enterprises (SME) is particularly welcome."
        }

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

            # Create a robust chain with the fixing parser and automatic retries
            # on transient errors like server overload or network issues.
            chain = (prompt | llm | output_fixing_parser).with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True,
                retry_if_exception_type=(groq.APIStatusError, requests.exceptions.RequestException)
            )
            
            # Initialize the retriever components once per evaluation run
            embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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

                        # results.append({
                        #     'file': gt_file,
                        #     'field': key,
                        #     'criterion': criterion,
                        #     'score': score,
                        #     'explanation': evaluation_result.explanations.get(criterion, "")
                        # })
                except OutputParserException as e:
                    # Catch parsing errors specifically if the LLM fails to produce valid JSON
                    print(f"    - Error parsing LLM output for key '{key}': {e}")
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

    # # Save the scores to a CSV file
    # if results:
    #     new_results_df = pd.DataFrame(results)
        
    #     # Combine with existing results if any
    #     if existing_df is not None:
    #         final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
    #     else:
    #         final_df = new_results_df

    #     os.makedirs(os.path.dirname(csv_output_filepath), exist_ok=True)
    #     final_df.to_csv(csv_output_filepath, index=False, encoding='utf-8')
    #     print(f"\nLLM-as-a-judge evaluation saved to {csv_output_filepath}")

    end = time.time()
    print(f"Script completed in {end-start} seconds.")


### Main function
if __name__ == '__main__':

    # Compute BLEU, ROUGE and BertScore metrics
    gt_path = r'.\evaluation\ground_truth'
    gen_path = r'.\evaluation\generated_outputs\gemini-2.5-flash'
    output_filepath = r'.\evaluation\metrics\gemini-2.5-flash\metrics.csv'
    # compute_metrics(gt_path, gen_path, output_filepath)

    # Plot boxplots and violin plots of metrics per json field
    output_folder_path = r'.\evaluation\plots\gemini-2.5-flash'
    #plot_metrics(output_filepath, output_folder_path)

    # Initialize the Groq LLM as the judge
    llm_judge = "llama3-70b-8192"
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found among the environment variables defined in .env")
    print("Evaluating with Llama 3 70B on Groq...")
    groq_llm = ChatGroq(
        model=llm_judge,
        temperature=0.1, 
    )

    llm_as_a_judge_evaluation(
        gt_path=r'.\evaluation\ground_truth',
        gen_path=r'.\evaluation\generated_outputs\gemini-2.5-flash',
        text_path=r'.\evaluation\data',
        csv_output_filepath=r'.\evaluation\llm_as_judge\gemini-2.5-flash_judged_by_'+llm_judge+'.csv',
        llm=groq_llm
    )

    # # Usage of LLM as a judge 
    # load_dotenv()
    # # --- Example 1: Using OpenAI's GPT-4o ---
    # print("Evaluating with GPT-4o...")
    # openai_llm = ChatOpenAI(
    #     model="gpt-4o", 
    #     temperature=0.1, 
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )

    # llm_as_a_judge_evaluation(
    #     gt_path="path/to/your/ground_truth_data",
    #     gen_path="path/to/your/generated_data",
    #     output_filepath="results_openai.csv",
    #     llm=openai_llm
    # )

    # # --- Example 2: Using a local model with LlamaCpp ---
    # print("Evaluating with LlamaCpp...")
    # # Make sure you have llama-cpp-python installed and a model file downloaded
    # llamacpp_llm = ChatLlamaCpp(
    #     model_path="/path/to/your/local-model.gguf",
    #     temperature=0.1,
    #     n_ctx=2048,
    #     # Add other necessary parameters for LlamaCpp
    # )

    # llm_as_a_judge_evaluation(
    #     gt_path="path/to/your/ground_truth_data",
    #     gen_path="path/to/your/generated_data",
    #     output_filepath="results_llamacpp.csv",
    #     llm=llamacpp_llm
    # )