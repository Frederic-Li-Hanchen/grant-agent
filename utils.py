import yaml
from typing import Dict, Any, List, Tuple
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
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
        ('compute_spearman_correlations', 'llm_judge_scores_csv_path'),
        ('compute_spearman_correlations', 'human_scores_csv_path'),
        ('compute_spearman_correlations', 'output_csv_path'),
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Get the variables and format the rest of the configuration
    formatted_config = format_config(config)
    return formatted_config


### Function to compute a custom Spearman's correlation coefficient that handles the cases where one or both inputs are constant.
### Chosen convention: (rho, p) are (0,1) or (1,0) if respectively one or both input(s) are constant 
# Input argument:
# - [pd.DataFrame] group: a DataFrame group with 'score_human' and 'score_llm' columns
# Output argument:
# - [pd.Series]: a pandas series containing the Spearman correlation and p values.
def spearman_corr_custom(group: pd.DataFrame) -> pd.Series:
    """
    Computes the Spearman correlation for a group, handling cases where one
    or both inputs are constant.

    Args:
        group (pd.DataFrame): A DataFrame group with 'score_human' and
                              'score_llm' columns.

    Returns:
        pd.Series: A series containing the 'spearman_correlation' and 'p_value'.
    """
    human_scores = group['score_human'].to_numpy()
    llm_scores = group['score_llm'].to_numpy()

    # Check for constant series
    is_human_constant = len(np.unique(human_scores)) == 1
    is_llm_constant = len(np.unique(llm_scores)) == 1

    if is_human_constant and is_llm_constant:
        # Both are constant and in perfect agreement on the (lack of) ranking.
        # Define this as a perfect correlation.
        return pd.Series({'spearman_correlation': 1.0, 'p_value': 0.0})
    elif is_human_constant or is_llm_constant:
        # One series is constant, the other is not. Correlation is undefined.
        # A reasonable convention is to return 0, as there can be no
        # co-variation if one variable does not vary.
        return pd.Series({'spearman_correlation': 0.0, 'p_value': 1.0})
    else:
        # Both series vary, calculate the standard Spearman correlation.
        correlation, p_value = spearmanr(human_scores, llm_scores)
        return pd.Series({'spearman_correlation': correlation, 'p_value': p_value})