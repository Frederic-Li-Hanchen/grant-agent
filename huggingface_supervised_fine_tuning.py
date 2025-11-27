import os
import json
import sys
import pandas as pd
from pdb import set_trace as st
from time import time
from dotenv import load_dotenv
from utils import load_config_from_yaml
from ctransformers import AutoModelForCausalLM as CTransformersAutoModel
import torch # Still needed for dtype conversion logic
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import warnings
# Filter warnings about torch and peft conflicts
warnings.filterwarnings("ignore", category=UserWarning, message=".*copying from a non-meta parameter.*")


### Function to compute the inference on the test set using HuggingFace and save them in the format required for evaluation (one json file with all fields per Bekanntmachung)
# Input:
# - [str] dataset_path: path to the input CSV dataset. The CSV must contain "file_path", "field", and "text" columns.
# - [str] model_name: name or path of the HuggingFace model to use for inference.
# - [str] output_dir: directory where the output JSON files will be saved.
# - [int] max_new_tokens: maximum number of new tokens to generate for each inference.
# - [bool] from_autotrain: whether the model to use was fine-tuned with AutoTrain or not 
# Output:
# - None
def get_huggingface_inferences(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 1024,
    autotrain_base_model_name: str = '',
) -> None:
    """
    Computes inferences for a test set using a HuggingFace model and saves the results.

    This function reads a CSV file where each row contains a prompt for the model.
    It runs inference for each prompt and groups the results by the original source
    file, saving one JSON file per source file in the specified output directory.

    Args:
        dataset_path (str): Path to the input CSV dataset.
                              The CSV must contain "file_path", "field", and "text" columns.
        model_name (str): The name or path of the HuggingFace model to use for inference.
                          For AutoTrain, this is the adapter model name.
        base_model_name (str): The name of the base model for AutoTrain adapters.
        output_dir (str): The directory where the output JSON files will be saved.
        max_new_tokens (int): The maximum number of new tokens to generate for each inference.
    """
    start_time = time()
    print(f"Starting inference process for model: {model_name}")

    # --- 1. Setup: Load model, tokenizer, and dataset ---
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    try:
        # Load the dataset from the CSV file
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} samples from {dataset_path}")

        # Check for required columns
        required_cols = ["file_path", "field", "text"]
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Dataset must contain the columns: {required_cols}")
            return

    except Exception as e:
        print(f"Error loading or validating dataset: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the quantized model using ctransformers in the case of a model not from AutoTrain
    from_autotrain = bool(autotrain_base_model_name)
    if not from_autotrain:
        try:
            print("Loading quantized model for inference...")
            # For CPU, we don't offload to GPU. For GPU, -1 means all layers.
            gpu_layers = 50 if torch.cuda.is_available() else 0
            model = CTransformersAutoModel.from_pretrained(
                model_name,
                model_type="mistral",
                gpu_layers=gpu_layers,
                # Increase context length if your prompts are very long
                context_length=8192 
            )
            print(f"Successfully loaded quantized model '{model_name}'.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            return
    else: # The model was fine-tuned with AutoTrain, requiring base model + LoRA adapter
        try:
            print(f"Loading base model '{autotrain_base_model_name}' for AutoTrain adapter...")
            # Load the base model and tokenizer from the transformers library
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # --- Critical fix for RecursionError ---
            # Set the padding token to the EOS token if it's not already set.
            # This is crucial for the model.generate() function to have a proper stopping criterion.
            tokenizer.pad_token = tokenizer.eos_token
            
            # Try loading with bfloat16 first, as it's common for modern LLMs and LoRA adapters
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    autotrain_base_model_name,
                    device_map='auto',
                    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance and compatibility
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Could not load with bfloat16 ({e}). Trying with float16...")
                # Fallback to float16 if bfloat16 is not supported or causes issues
                model = AutoModelForCausalLM.from_pretrained(
                    autotrain_base_model_name,
                    device_map='auto',
                    torch_dtype=torch.float16, # Fallback to float16
                    trust_remote_code=True
                )

            print(f"Loading and applying LoRA adapter '{model_name}'...")
            # Apply the LoRA adapter to the base model
            peft_model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
            print("Successfully loaded AutoTrain model.")

        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            return

    # Define a custom stopping criteria class to prevent recursion errors
    class EosTokenStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_token_id: int):
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            # Check if the last generated token is the EOS token
            last_token = input_ids[0, -1].item()
            return last_token == self.eos_token_id

    # --- 2. Inference Loop ---
    # Group samples by the original file path to create one JSON per file
    for file_path, group in df.groupby("file_path"):
        base_filename = os.path.basename(str(file_path).replace("\\", "/")) # Normalize path separators for cross-platform compatibility
        output_filename = os.path.splitext(base_filename)[0] + ".json"
        output_filepath = os.path.join(output_dir, output_filename)
        results_for_file = {}
        # If an output file already exists, load it to resume from where we left off.
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, 'r', encoding='utf-8') as f:
                    results_for_file = json.load(f)
                print(f"Resuming for {base_filename}, loaded existing results.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {output_filepath}. Overwriting.")
                results_for_file = {}

        print(f"Processing file: {base_filename}")

        # Iterate over each field (objective, deadline, etc.) for the current file
        for _, row in group.iterrows():
            field = row["field"]
            # Skip inference for this field if it already exists in the results
            if field in results_for_file:
                print(f"  - Skipping field '{field}', already exists in output.")
                continue

            # The prompt is the "human" part of the text
            prompt = row["text"].split("\n bot:")[0]

            if not from_autotrain:
                # Generate the output from the GGUF model
                output = model(prompt, max_new_tokens=max_new_tokens, stop=["human:"], stream=False)
                results_for_file[field] = output.strip()
            else:
                # Generate output for the transformers-based AutoTrain model
                # The stop argument is handled differently here
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Create a stopping criteria list
                stopping_criteria = StoppingCriteriaList([EosTokenStoppingCriteria(tokenizer.eos_token_id)])

                output_ids = peft_model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id # Keep this for padding
                )
                response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                results_for_file[field] = response.strip()

            print(f"  - Generated answer for field: '{field}'")
            
            # --- 3. Save Results Incrementally ---
            # Save the results after each field is processed to make the script more robust.
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(results_for_file, f, indent=4, ensure_ascii=False)
            print(f"  -> Updated {output_filepath} with result for '{field}'.")

    end_time = time()
    print(f"\nInference process completed in {end_time - start_time:.2f} seconds.")



### Main function
if __name__ == "__main__":
    ### Load the YAML config file and main logic
    try:
        config = load_config_from_yaml(r'./config.yaml')
    except:
        print("ERROR: YAML config file './config.yaml' not found!")
        sys.exit(1)

    run_steps = config.get('run_steps', {})

    ### HuggingFace model inference
    if run_steps.get('compute_huggingface_inferences'):
        print("\n--- Running HuggingFace LLM inferences ---")
        params = config.get('compute_huggingface_inferences',{})

        # For CPU inference, it's highly recommended to use a quantized GGUF model.
        # The original model name is kept for reference, but we'll use a GGUF version.
        model_name = params.get('model_name')
        
        if model_name and 'GGUF' in model_name.upper():
            print(f"Using quantized GGUF model for efficiency: {model_name}")

        get_huggingface_inferences(
            dataset_path=params['dataset_path'],
            model_name=params.get('model_name'),
            output_dir=params['output_dir'],
            max_new_tokens=params['max_new_tokens'],
            autotrain_base_model_name=params.get('autotrain_base_model_name'),
        )
