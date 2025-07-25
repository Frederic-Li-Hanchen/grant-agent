### grant-agent
Agent retrieving key information from grant calls issued by the German Federal Ministries (e.g. BMFTR, BMWE, etc.)

Repository contents:
- build_database.py: script to scrape a list of cleaned text data files containing Bekanntmachungen (currently only from the BMFTR website)
- check_databased_quality.py: script to check the quality of the text data files formatting 
- grant_summarisation_agent.py: script to extract specific key information from a grant document (currently only in PDF or txt format)
