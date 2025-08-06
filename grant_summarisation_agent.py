from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json
from dotenv import load_dotenv
from time import time
#import pytesseract
from pdb import set_trace as st


### Extract information from a document containing a grant call. API key is expected to be contained in a .env file.
# Input arguments:
# - [str] doc_path: path to the document (currently either PDF or text)
# - [str] output_path: path to the folder where the resulting json file will be saved. If left empty, the file is not saved.
# - [int] chunk_size: size of the text chunks to be created from the document
# - [int] chunk_overlap: overlap between the text chunks
# Output:
# - [dict] extracted_info: structured information extracted from the document
def extract_info_from_document(doc_path,output_path='',chunk_size=4000,chunk_overlap=200): # NOTE: chunk size and overlap have to be somewhat tuned depending on the document size and content
    """
    Information to be extracted:
    - Main theme and objectives of the call
    - Inclusion and exclusion criteria (e.g. de-minimis rule, whether industrial partners are required, whether research institutions are allowed, etc.)
    - Deadline of the submission
    - Maximum funding allowed
    - Maximum duration of the project allowed
    - Information about the procedure to follow for the submission
    - Contact person(s) information for further questions if required
    - Any other relevant information that could be useful for the grant application process
    """
    start = time()

    # Check if the generated output already exists. If yes, skip to processing.
    if len(output_path) > 0:
        # Create an appropriate file name by retrieving the original document name and chaging the extension to .json
        file_name = os.path.basename(doc_path)
        if '.pdf' in file_name:
            new_file_name = file_name.replace('.pdf','.json')
        else:
            new_file_name = file_name.replace('.txt','.json')
        if os.path.isfile(os.path.join(output_path,new_file_name)):
            print(f"Output file {new_file_name} already exists. Skipping processing for {doc_path}.")
            with open(os.path.join(output_path,new_file_name),'r',encoding='utf-8') as f:
                extracted_info = json.load(f)
            return extracted_info

    # Load the document containing the grant (PDF or text)
    if doc_path.endswith('.pdf'):
        print(f"Loading PDF document from {doc_path}...")
        loader = PyPDFLoader(doc_path) 
        # # NOTE: this is only required for the extraction from complex PDF documents for which PyPDFLoader would fail
        # os.environ["OCR_AGENT"] = "tesseract" # Set the OCR agent to use Tesseract for OCR processing
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Path to the Tesseract executable
        #loader = UnstructuredPDFLoader(doc_path) # TODO: fix a problem with the pytesseract library causing this to fail. 

    elif doc_path.endswith('.txt'):
        print(f"Loading text document from {doc_path}...")
        loader = TextLoader(doc_path)

    else:
        raise ValueError("The provided path is neither a PDF or text document. Please provide a valid file path.")
    
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #split_docs = text_splitter.split_documents(documents)

    # Initialise the LLM
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY not found among the environment variables defined in .env")
    os.environ['GOOGLE_API_KEY'] = gemini_api_key
    llm = init_chat_model("gemini-2.5-flash", temperature=0.1, max_tokens=4000, model_provider='google_genai') 

    # # Detect language of the document and translate it into English if necessary
    # translation_query = ChatPromptTemplate.from_template(
    #     "If the document is not in English, translate it into English. Provide only the translation. If already in English, return it as is.\n\n{text}"
    #     )
    # translation_chain = LLMChain(llm=llm, prompt=translation_query, output_key="english_text", verbose=True)
    # translated_chunks = []
    # for chunk in split_docs:
    #     translation = translation_chain({"text": chunk.page_content})
    #     translated_chunks.append(translation["english_text"])

    # Information extraction
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # NOTE: other embeddings could be experimented with
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = loader.load()
    split_docs = text_splitter.split_documents(documents)
    index = DocArrayInMemorySearch.from_documents(split_docs, embeddings)
    #index = VectorstoreIndexCreator(embedding=embeddings,vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

    # Define multiple smaller prompts to avoid causing suboptimal matching between the prompt and chunks embeddings
    # NOTE: one startegy to test could be to first ask the LLM to identify the correct sections, then extract the information from them.
    objective_query = "From the provided text, identify the key objectives of the call and summarise them in a few (e.g. 2 to 5) sentences.\n\
        This information is usually found in the first sections of the document.\n\
        Focus in particular on the following aspects (if they can be found):\n\
        what should be developped in this project?\n\
        for whom, i.e. which target groups?\n\
        how should it be developped?\n\
        are there specific examples of topics that the call focuses on?" 
    # NOTE: this one is not so easy to define, because the goals can be very broad and vary from call to call.
    inclusion_query = "From the provided text, list any inclusion criteria for applicants to the call. Check more specifically the following information:\n\
        what are the eligible applicants (e.g. university, research institution, company, etc.)?\n\
        are there any specific requirements for the applicants (e.g. past experience in a specific field)?\n\
        If nothing about inclusion criteria can be found, return None.\n\
        Example: Der Verbund muss aus mindestens (Minimalanforderung) zwei deutschen Partnern (eine Hochschule oder außer­universitäre Forschungseinrichtung \
        und ein Unternehmen der gewerblichen Wirtschaft – insbesondere KMU) und drei kanadischen Partnern \
        (ein Forschungszentrum, eine Universität und ein zuwendungsfähiger kanadischer Firmenpartner) bestehen (2 + 3-Bekanntmachung).\n\
        Expected answer: The consortium must have 5 partners: one university or research institution and one company from Germany, and one research institution, one university and one company from Canada."
    exclusion_query = "From the provided text, list any exclusion criteria that could apply to applicants to the call. Check more specifically for the following information:\n \
        is any type of institution prohibited from applying (e.g. companies, research institutions, etc.)? \
        do applicants have to come from a specific region of the world (e.g. Europe, a specific country)?\
        If none about exclusion criteria can be found, return None.\n\
        Example: Hochschulen, die keine Forschungseinrichtungen sind, sind im Rahmen dieser Ausschreibung nicht förderfähig.\n\
        Expected answer: Universities that are not research institutions are excluded from the call."    
    deadline_query = "From the provided text, return the latest time at which the first stage of the application can be submitted. \
        Return only the date formatted as DD.MM.YYYY (and HH:MM if available).\n\
        If several deadlines are provided, return them in the aforementioned format, separated by semi-colons.\n\
        If no deadline can be found, return None.\n\
        Example: In der ersten Verfahrensstufe sind dem ABC Projektträger\
        bis spätestens 25. November 2025 zunächst Projektskizzen in schriftlicher\
        und/oder elektronischer Form über „online-submission-platform“ vorzulegen.\n\
        Expected answer: 25.11.2025"
    max_funding_query = "From the provided text, return the maximum funding allowed for the project in euros (or dollars if applicable).\n\
        Use comma as thousand separator and dot as decimal point, with the associated currency (e.g. euros, dollars).\n\
        If the information is available, specify if the amount applies to the whole consortium or per partner.\n\
        Be careful as some amounts that are not the maximum funding amount may also be listed in the text.\n\
        If no amount can be found, return Not specified.\n\
        Example: Die Zuwendungen werden im Wege der Projektförderung als nicht rückzahlbarer Zuschuss und in der Regel mit bis zu 650 000 Euro pro deutschem Forschungsverbund.\n\
        Darüber hinaus veröffentlicht das Ministerium Fördermittel über 100 000 Euro in der Transparenzdatenbank der Ethik-Kommission.\n\
        Expected answer: 650,000 euros per partner" # NOTE: add one more example where project/partner information is not specified?
    max_duration_query = "From the provided text, return the maximum duration of the project allowed in months.\n\
        Be careful as some durations that are not the maximum duration allowed for the project may appear in the text.\n\
        If none can be found, return Not specified.\n\
        Example: Die Zuschüsse werden als nicht rückzahlbare Projektförderung in der Regel für einen Zeitraum von 24 bis 36 Monaten vergeben.\n\
        Im Fall der Zweitveröffentlichung soll die Embargofrist zwölf Monate nicht überschreiten.\n\
        Expected answer: 36 months"
    procedure_query = "From the provided text, list any concrete instruction regarding the grant application process, such as documents to prepare, steps to follow.\n\
        Specify the number of stages of the procedure (usually 1, 2 or 3).\n\
        Pay attention to submission details regarding the documents to be submitted for the first stage,\
        e.g. language, maximum number of pages allowed, whether the document must be submitted electronically or in written form.\n\
        If none can be found, return None."
    contact_query = "From the provided text, return information about the person(s)\
        that can be contacted for further questions, including whenever possible name, e-mail and phone number, separated by commas.\n\
        If multiple contact persons can be found, return them all in the same format, separated by semicolons.\n\
        Note that multiple contact persons may share the same contact information.\n\
        If none can be found, return None.\n\
        Example: Ansprechpersonen sind:\n\
        Frau Alice Schmidt\n\
        Telefon: +49 123 456789\n\
        E-Mail: alice.schmidt@bmftr.de\n\
        Herr Robert Müller\n\
        Telefon: +49 987 654321\n\
        E-Mail: robert.mueller@bmftr.de\n\
        Expected answer: Alice Schmidt, alice.schmidt@bmftr.de, +49 123 456789; Robert Müller, robert.mueller@bmftr.de, +49 987 654321"
    misc_query = "From the provided text, return any other relevant information that could be useful for the grant application process.\n\
        Examples of such information include whether the de-minimis rule applies, Förderquote for different partners, whether specific \
        institutions are expressly invited to apply, whether geographical restrictions apply to the results of the project, etc.\n\
        If none can be found, return None."
    
    # # Obtain the results for each query
    # objective_results = index.query(objective_query, llm=llm, return_source_documents=False, verbose=True)
    # inclusion_results = index.query(inclusion_query, llm=llm, return_source_documents=False, verbose=True)
    # exclusion_results = index.query(exclusion_query, llm=llm,return_source_documents=False, verbose=True)
    # deadline_results = index.query(deadline_query, llm=llm, return_source_documents=False, verbose=True)
    # max_funding_results = index.query(max_funding_query, llm=llm, return_source_documents=False, verbose=True)
    # max_duration_results = index.query(max_duration_query, llm=llm, return_source_documents=False, verbose=True)
    # procedure_results = index.query(procedure_query, llm=llm, return_source_documents=False, verbose=True)
    # contact_results = index.query(contact_query, llm=llm, return_source_documents=False, verbose=True)
    # misc_results = index.query(misc_query, llm=llm, return_source_documents=False, verbose=True)
    
    # Create a RetrievalQA chain to query the index
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=index.as_retriever(),
        return_source_documents=False,
        verbose=True,
    )

    # Obtain the results for each query and build the final dictionary
    queries = {
        "objective": objective_query,
        "inclusion_criteria": inclusion_query,
        "exclusion_criteria": exclusion_query,
        "deadline": deadline_query,
        "max_funding": max_funding_query,
        "max_duration": max_duration_query,
        "procedure": procedure_query,
        "contact": contact_query,
        "misc": misc_query,
    }

    extracted_info = {
        key: qa_chain.invoke({"query": query})["result"] for key, query in queries.items()
    }

    # # DEBUG to retrieve the most relevant chunks retrieved given a specific query
    # retriever = index.vectorstore.as_retriever()
    # t = retriever.get_relevant_documents(deadline_query)
    # t1 = retriever.get_relevant_documents(contact_query)
    # st()

    ### TODO: implementing a post-processing routine to make sure the LLM outputs follow the expected format may be required.

    # Save the result dictionary into a json file if enabled
    if len(output_path) > 0:

        with open(os.path.join(output_path,new_file_name),'w',encoding='utf-8') as f:
            json.dump(extracted_info,f,indent=4,ensure_ascii=False)
    
    end = time()
    print('Information extraction for document %s completed in %.2f seconds.' % (doc_path,end-start))

    return extracted_info



### Extract information from webpage
### TODO: implement once the information extraction agent from text is properly working
def extract_info_from_webpage(webpage_url):
    """
    Steps to be performed:
    - Load webpage content
    - Detect language of the content
    - Translate into English if necessary
    - Split the text into chunks
    - Extract the required information from the chunks
    - Save and return the extracted information in a structured format

    Information to be extracted:
    - Main theme and objectives of the call
    - Inclusion and exclusion criteria (e.g. de-minimis rule, whether industrial partners are required, whether research institutions are allowed, etc.)
    - Deadline of the submission
    - Maximum funding allowed
    - Maximum duration of the project allowed
    - Information about the procedure to follow for the submission
    - Contact person(s) information for further questions if required
    - Any other relevant information that could be useful for the grant application process
    """
    pass  # Implementation goes here




### Main function
if __name__ == "__main__":

    # Hyper-parameters
    data_folder = r'C:\Users\Frederic\Documents\Programming\Grant-agent\evaluation\data'
    output_path = r'C:\Users\Frederic\Documents\Programming\Grant-agent\evaluation\generated_outputs'
    chunk_size = 4000
    chunk_overlap = 200

    # Loop on the examples with associated ground truth
    doc_list = [e for e in os.listdir(data_folder) if e.endswith('txt')]
    for doc_name in doc_list:
        doc_path = os.path.join(data_folder,doc_name)
        extract_info_from_document(doc_path=doc_path,output_path=output_path,chunk_size=chunk_size,chunk_overlap=chunk_overlap)