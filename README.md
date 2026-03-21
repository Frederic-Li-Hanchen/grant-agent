### grant-agent
Agent retrieving key information from grant calls (Bekanntmachungen) issued by German Federal Ministries (e.g. BMFTR, BMWK, etc.)

---

### Repository structure

**`agent/`** — production agent implementation
- `agent.py`: main entrypoint (`python agent/agent.py --input mail.txt --output results.xlsx`)
- `fetcher.py`: link identification and call text fetching (HTML scraping and PDF download)
- `extractor.py`: RAG-based extraction of key fields from call documents
- `exporter.py`: Excel export of extracted results

**`research/`** — pilot study: comparative evaluation of RAG approaches on BMBF Bekanntmachungen
- `build_database.py`: scrape and preprocess Bekanntmachungen text files; build vector and graph databases
- `check_database_quality.py`: check formatting quality of scraped text files
- `grant_summarisation_agent.py`: vector RAG and graph RAG extraction (Gemini and fine-tuned Mistral)
- `create_ground_truth.py`: helper script to create ground truth for fine-tuning and evaluation
- `evaluate_agent.py`: compute evaluation metrics (BLEU, ROUGE, BERTScore, BARTScore) and plot results
- `huggingface_supervised_fine_tuning.py`: fine-tune a LLM on annotated data using HuggingFace
- `config.yaml`: configuration parameters for the research pipeline

**`prompts/`** — shared JSON prompt templates (used by both research and agent)

**`notes/`** — project documentation
- `agent_specifications.txt`: specifications and work plan for the agent implementation

**`utils.py`** — shared helper functions (config loading, web scraping, etc.)
**`pyproject.toml` / `uv.lock`** — Python dependency configuration

---

### Usage

1. Save the newsletter email as a plain-text file (e.g. `mail.txt`).
2. Set your Gemini API key in a `.env` file at the repo root:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Run the agent:
   ```
   python agent/agent.py --input mail.txt --output results.xlsx
   ```
   The agent identifies funding-call links in the email, fetches each call, extracts
   the key fields via RAG, and writes the results to `results.xlsx`.

The LLM provider and all other parameters are configured in `config.yaml`.

---

### Current status
A comparative study of RAG approaches (vector RAG with Gemini-2.5-Flash and fine-tuned Mistral-7B-Instruct, graph RAG) was carried out on a dataset of BMBF Bekanntmachungen. Vector RAG with Gemini-2.5-Flash was identified as the most promising approach for the production agent, followed closely by the fine-tuned Mistral-7B-Instruct approach.
The preliminary agent implementation is complete.
