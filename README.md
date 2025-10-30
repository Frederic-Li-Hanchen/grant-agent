### grant-agent
Agent retrieving key information from grant calls issued by the German Federal Ministries (e.g. BMFTR, BMWE, etc.)

Repository contents:
- config.yaml: config file of parameters 
- build_database.py: script to scrape a list of cleaned text data files containing Bekanntmachungen (currently only from the BMFTR website)
- check_database_quality.py: script to check the quality of the text data files formatting 
- grant_summarisation_agent.py: script to extract specific key information from a grant document (currently only in PDF or txt format)
- create_ground_truth.py: helper script to create a ground truth for fine-tuning and evaluation
- evaluate_agent.py: script to evaluate the outputs produced by the agent using various evaluation strategies and plot the results 
- huggingface_supervised_fine_tuning.py: script to fine-tune a LLM on annotated data using HuggingFace
- utils.py: script containing helper functions
- ./prompts/: folder containing json files with template information for prompts 
