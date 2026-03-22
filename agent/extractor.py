"""
RAG-based information extraction.

Responsibilities:
- Preprocess call text (chunking).
- Run vector-based RAG to retrieve each target field.
- Support google_genai (Gemini) and huggingface (fine-tuned Mistral) providers.
- Ministry-specific prompt templates via config (falls back to default prompts).
"""
import json
import os
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type


def build_llm_and_embeddings(config: dict[str, Any]) -> tuple:
    """
    Initialise and return (llm, embeddings, retry_on) once for reuse across documents.

    Separating initialisation from per-document extraction avoids reloading a
    large model for every call.

    Returns:
        (llm, embeddings, retry_on) where retry_on is a tenacity retry condition.
    """
    load_dotenv()

    llm_cfg = config.get('llm', {})
    model_provider: str = llm_cfg.get('model_provider', 'google_genai')
    model_name: str = llm_cfg.get('model_name', 'gemini-2.5-flash')
    temperature: float = llm_cfg.get('temperature', 0.1)
    max_tokens: int = llm_cfg.get('max_tokens', 4000)

    if model_provider == 'google_genai':
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from google.api_core.exceptions import ResourceExhausted

        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key is None:
            raise ValueError('GEMINI_API_KEY not found in environment variables')
        google_embedding_model: str = llm_cfg.get('google_embedding_model', 'models/gemini-embedding-001')
        llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature,
            max_tokens=max_tokens, google_api_key=gemini_api_key)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=google_embedding_model, google_api_key=gemini_api_key)
        retry_on = retry_if_exception_type(ResourceExhausted) | retry_if_exception_type(Exception)

    elif model_provider == 'huggingface':
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
        from peft import PeftModel
        from langchain_community.llms import HuggingFacePipeline
        from langchain_community.embeddings import HuggingFaceEmbeddings

        autotrain_base_model_name: str = llm_cfg.get('autotrain_base_model_name', '')
        embedding_model_name: str = llm_cfg.get('embedding_model_name', 'paraphrase-multilingual-mpnet-base-v2')

        if not autotrain_base_model_name:
            raise ValueError('autotrain_base_model_name must be set in config when model_provider is huggingface')

        print(f'  Loading base model {autotrain_base_model_name!r} ...')
        hf_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Use device_map='cuda:0' rather than 'auto' to avoid layer offloading to
        # CPU/disk, which causes merge_and_unload() to fail on meta-device parameters.
        device_map = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        base_model = AutoModelForCausalLM.from_pretrained(
            autotrain_base_model_name, torch_dtype=hf_dtype, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f'  Applying LoRA adapter {model_name!r} ...')
        peft_model = PeftModel.from_pretrained(base_model, model_name)
        peft_model = peft_model.merge_and_unload()

        pipe = hf_pipeline(
            'text-generation', model=peft_model, tokenizer=tokenizer,
            max_new_tokens=max_tokens, temperature=float(temperature),
            do_sample=bool(temperature > 0))
        llm = HuggingFacePipeline(pipeline=pipe)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        retry_on = retry_if_exception_type(Exception)

    else:
        raise ValueError(f"Unsupported model_provider {model_provider!r}. Choose 'google_genai' or 'huggingface'.")

    return llm, embeddings, retry_on


def extract_fields(call_text: str, llm: Any, embeddings: Any, retry_on: Any, config: dict[str, Any]) -> dict:
    """
    Extract target fields from a call document using vector RAG.

    Args:
        call_text:  Plain text of the funding call.
        llm:        Pre-built LangChain LLM (from build_llm_and_embeddings).
        embeddings: Pre-built LangChain embeddings (from build_llm_and_embeddings).
        retry_on:   Tenacity retry condition (from build_llm_and_embeddings).
        config:     Agent configuration dict (from config.yaml).

    Returns:
        Dict with keys: objective, inclusion_criteria, exclusion_criteria,
        deadline, max_funding, max_duration, procedure, contact, misc.
    """
    from langchain_classic.chains import RetrievalQA
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import DocArrayInMemorySearch

    rag_cfg = config.get('rag', {})
    chunk_size: int = rag_cfg.get('chunk_size', 4000)
    chunk_overlap: int = rag_cfg.get('chunk_overlap', 200)
    prompts_filepath: str = rag_cfg.get('prompts_filepath', 'prompts/agent_prompts.json')

    # --- Build vector index from call text ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.create_documents([call_text])
    index = DocArrayInMemorySearch.from_documents(split_docs, embeddings)

    # --- Load prompts ---
    with open(prompts_filepath, 'r', encoding='utf-8') as f:
        queries: dict[str, str] = json.load(f)

    # --- RetrievalQA chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=index.as_retriever(),
        return_source_documents=False,
        verbose=False,
    )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_on,
    )
    def _invoke(query: str) -> str:
        return qa_chain.invoke({'query': query})['result']

    # --- Extract each field ---
    extracted_info = {key: _invoke(query) for key, query in queries.items()}
    return extracted_info
