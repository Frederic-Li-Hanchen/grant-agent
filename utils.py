import yaml
from typing import Dict, Any, List, Tuple
from pdb import set_trace as st

### Helper function to format strings in a config dictionary
# Input arguments:
# - [Dict[str, Any]] config: configuration dictionary containing variable placeholders
# - [Dict[str, Any]] variables: values of the variables to use
# Output argument:
# - [Dict[str, Any]] config: the formatted configuration dictionary
def format_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extract variables for formatting
    variables = config.get('vars', {})
    if not variables:
        return config  # No variables to format with, return as is.

    # 2. Define a list of key-paths to format. This makes it easy to manage.
    # Note: Even with YAML aliases, we must format each path string individually
    # because strings are immutable in Python.
    paths_to_format: List[Tuple[str, ...]] = [
        ('evaluation_shared_paths', 'gen_path'),
        ('evaluation_shared_paths', 'traditional_metrics_csv_file'),
        ('evaluation_shared_paths', 'llm_scores_csv_filepath'),
        ('compute_traditional_metrics', 'gen_path'),
        ('compute_traditional_metrics', 'output_filepath'),
        ('plot_traditional_metrics', 'csv_filepath'),
        ('plot_traditional_metrics', 'output_folder_path'),
        ('run_llm_as_judge', 'gen_path'),
        ('run_llm_as_judge', 'csv_output_filepath'),
        ('compute_overall_llm_judge_scores', 'judge_scores_csv_path'),
        ('compute_overall_llm_judge_scores', 'output_csv_path'),
    ]

    for path_keys in paths_to_format:
        # Navigate to the nested dictionary if necessary
        sub_dict = config
        try:
            for key in path_keys[:-1]:
                sub_dict = sub_dict[key]
            # Format the string at the final key
            final_key = path_keys[-1]
            if final_key in sub_dict and isinstance(sub_dict[final_key], str):
                sub_dict[final_key] = sub_dict[final_key].format(**variables)
        except KeyError:
            # This path doesn't exist in the config, so we skip it.
            # You could add a warning here if a path is expected to always exist.
            # print(f"Warning: Path {path_keys} not found in config.")
            pass
            
    return config

### Helper function to load the YAML configuration file
# Input argument:
# - [str] config_path: the path to the YAML config file
# Output argument:
# - [Dict[str, Any]] config: the configuration dictionary
def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Get the variables and format the rest of the configuration
    formatted_config = format_config(config)
    return formatted_config