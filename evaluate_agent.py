import evaluate
import os
import bert_score
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
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
    # violin_output_filepath = output_filepath.replace('boxplot', 'violinplot')
    # if violin_output_filepath == output_filepath:  # if 'boxplot' not in name
    #     name, ext = os.path.splitext(output_filepath)
    #     violin_output_filepath = f"{name}_violin{ext}"

    plt.savefig(os.path.join(output_folder_path,'violinplot.png'), bbox_inches='tight')
    print(f"Violin plot saved to {output_folder_path}")
    plt.close(fig)  # Close the figure to free memory



### Main function
if __name__ == '__main__':

    # Compute BLEU, ROUGE and BertScore metrics
    # gt_path = r'.\evaluation\ground_truth'
    # gen_path = r'.\evaluation\generated_outputs\gemini-2.5-flash'
    output_filepath = r'.\evaluation\metrics\gemini-2.5-flash\metrics.csv'
    # compute_metrics(gt_path, gen_path, output_filepath)

    # Plot boxplots of metrics per json field
    output_folder_path = r'.\evaluation\boxplots\gemini-2.5-flash'
    plot_metrics(output_filepath, output_folder_path)