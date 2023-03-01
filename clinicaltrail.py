from pytrials.client import ClinicalTrials
import jmespath
import gradio as gr
import re
from openai.embeddings_utils import get_embedding, cosine_similarity
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import GPT2Tokenizer
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from gpt_index import Document
import tiktoken
import openai
import uuid
import numpy as np
import pandas as pd
import os
import tqdm

openai.api_type = "azure"
openai.api_base = 'https://openai-demo-endpoint-we.openai.azure.com/'
openai.api_version = "2022-12-01"
openai.api_key = '79e8b296c36447919d942f9b17877b72'
redis_key = '2Ek0dViRcVV77ERAsZET+kuzhGZkRdkkKIKaIYhzbJ4='
redis_endpoint = 'redissearch.southcentralus.redisenterprise.cache.azure.net'

# Redis configuration
#Ada: 1536 dimensions
#Babbage: 2048 dimensions
#Curie: 4096 dimensions
#Davinci: 12288 dimensions
DIM = 1536
VECT_NUMBER = 3155
index_name = "embeddings-index"
url_prefix = "https://clinicaltrials.gov/ct2/show/"

# Connect to the Redis server
redis_conn = Redis(host= redis_endpoint, port=10000, password=redis_key, ssl=True)

doc_engine = 'text-embedding-ada-002'
query_engine = 'text-embedding-ada-002'
qna_engine = 'text-davinci-003'
encoding_name = "cl100k_base"  # For second-generation embedding models like text-embedding-ada-002, use the cl100k_base encoding.
top_n_results = 2

# Connect to the Redis server
redis_conn = Redis(host= redis_endpoint, port=10000, password=redis_key, ssl=True)

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def get_clinical_study(study_name , num_studies):
    ct = ClinicalTrials()
    desc = ct.get_full_studies(search_expr=study_name, max_studies=num_studies)
    detail_lst = []    

    for i in range(num_studies):
        detail_lst.append([])
        NCTId = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.IdentificationModule.NCTId',desc)
        source_url = url_prefix + NCTId
        RecruitmentStatus = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.StatusModule.OverallStatus',desc)
        #print (RecruitmentStatus)

        BriefSummary = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.BriefSummary',desc)
        DetailedDescription = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.DetailedDescription',desc)
        EligibilityCriteria = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.EligibilityModule.EligibilityCriteria',desc)
        
        EligibilityCriteria = normalize_text(EligibilityCriteria)

        BriefSummary = normalize_text(BriefSummary)
        
        detail_lst[i].append(source_url)
        detail_lst[i].append(EligibilityCriteria)

        '''
        if DetailedDescription is None:
            BriefSummary = normalize_text(BriefSummary)
            detail_lst[i].append(source_url)
            detail_lst[i].append(BriefSummary + EligibilityCriteria)
        else:
            DetailedDescription = normalize_text(DetailedDescription)
            BriefSummary = normalize_text(BriefSummary)
            detail_lst[i].append(source_url)
            detail_lst[i].append(BriefSummary + DetailedDescription + EligibilityCriteria)
        '''
    return detail_lst

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))

def chunk_and_embed(text, source_url):

    full_data = {
       "source_url" : None,
        "text": None,
        "doc_embeddings": None
    }

    # initialize a text splitter (why? GPT-3 has a limited context window, so we need a way to chunk our documents and pass in the relevant content)
    
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len)
    
    chunks = text_splitter.split_text(text)

    #documents = []

    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk, engine = doc_engine)

        full_data['source_url'] = source_url
        full_data['text'] = text
        full_data['doc_embeddings'] = emb

        #doc = Document(chunk, embedding= emb, doc_id=f"doc_id_{i}", extra_info = source_url)
        #documents.append(doc)
    return full_data

def delete_index(index):
    result = redis_conn.execute_command("FT.DROP", index)  
    print('Index Deleted.')

def create_index(redis_conn: Redis, index_name="embeddings-index", prefix = "embedding",number_of_vectors = VECT_NUMBER, distance_metric:str="COSINE"):
    source_url = TextField(name="source_url")
    text = TextField(name="text")
    embeddings = VectorField("embeddings",
                "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": DIM,
                    "DISTANCE_METRIC": distance_metric,
                    "INITIAL_CAP": number_of_vectors,
                })
    # Create index
    redis_conn.ft(index_name).create_index(
        fields = [source_url, text, embeddings],
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    )

def set_document(elem):
    index = str(uuid.uuid4())
    redis_conn.hset(
        f"embedding:{index}",
        mapping={
            "source_url": elem['source_url'],
            "text": elem['text'],
            "embeddings": np.array(elem['doc_embeddings']).astype(dtype=np.float32).tobytes()
        }
    )

def add_embeddings(source_url, text):
    embeddings = chunk_and_embed(source_url, text)
    if embeddings:
        # Store embeddings in Redis
        set_document(embeddings)

def get_documents(number_of_results: int=VECT_NUMBER):
    base_query = f'*'
    return_fields = ['id', 'source_url', 'text']
    query = Query(base_query)\
        .paging(0, number_of_results)\
        .return_fields(*return_fields)\
        .dialect(2)
    results = redis_conn.ft(index_name).search(query)
    if results.docs:
        return pd.DataFrame(list(map(lambda x: {'id' : x.id, 'source_url': x.source_url, 'text': x.text}, results.docs)))
    else:
        return pd.DataFrame()


def execute_query(np_vector:np.array, return_fields: list=[], search_type: str="KNN", number_of_results: int=top_n_results, vector_field_name: str="embeddings"):
    base_query = f'*=>[{search_type} {number_of_results} @{vector_field_name} $vec_param AS vector_score]'
    query = Query(base_query)\
        .sort_by("vector_score")\
        .paging(0, number_of_results)\
        .return_fields(*return_fields)\
        .dialect(2)
    
    params_dict = {"vec_param": np_vector.astype(dtype=np.float32).tobytes()}

    results = redis_conn.ft(index_name).search(query, params_dict)
    return pd.DataFrame(list(map(lambda x: {'id': x.id, 'source_url': x.source_url, 'text': x.text, 'vector_score': x.vector_score}, results.docs)))

# Semantically search using the computed embeddings on RediSearch
def search_semantic_redis(search_query, pprint=True):
    embedding = get_embedding(search_query, engine = query_engine)
    res = execute_query(np.array(embedding))

    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res.reset_index()

def create_context(question):
    """
    Find most relevant context for a question via Redisearch
    """
    res = search_semantic_redis(question)
    num_tokens_from_string("tiktoken is great!", "gpt2")
    if len(res) == 0:
        return None, "No vectors matched, try a different context."
    
    res_text = "\nStudy Link: ".join(res['source_url'] + "\n" + "Criteria: " + res["text"] + "\n\n")
    n_tokens = num_tokens_from_string(res_text, encoding_name)

    # This model's maximum context length is 4097 tokens. So limiting the context within the length
    if n_tokens >3500:
        res_text = res_text[:16000]
    return (res_text)


def add_docs(study_name, number_of_studies):
    """
    Adding the base docuents' embeddings into Redisearch
    """
    # Check if Redis index exists
    index_name = "embeddings-index"
    #delete_index(index_name)
    try:
        if redis_conn.ft(index_name).info():
            print("Index exists")
    except:
        print("Index does not exist")
        print("Creating index")
        # Create index 
        create_index(redis_conn)

    studis = get_clinical_study(study_name, number_of_studies)

    for inner_list in studis:
        add_embeddings(inner_list[1],inner_list[0])


def main():
    search_query = 'A 30 years old woman not pragnent presents with symptoms of COVID-19, including fever, cough, fatigue, and loss of taste and smell. The patient reports that these symptoms began approximatel 7 days ago and have progressively worsened since onset. The patient reports that they recently returned from international travel and had close contact with someone diagnosed with COVID-19.'
    
    #search_query = 'A 30 years old patient who was recently diagnosed with clear cell sarcoma, a rare type of soft tissue cancer that most commonly affects the limbs. The tumor is localized and has not spread to other parts of the body. The patient has undergone imaging tests and a biopsy to confirm the diagnosis. Despite standard treatments, such as surgery and chemotherapy, the tumor has not responded as expected.'
   
    content = create_context(search_query)
    question = "Is this patient eligible for Clear cell sarcoma clinical trail?"
    #print(content)

    prompt_template = """The following is a conversation between doctor and a clinical trials matching bot.
The bot carefully evaluates below mentioned each clinical trials eligibility criteria to answer "Yes/No". Bot also provides the detail explanations and crosponding study names. 
If doctors asks a quesion in a different context then state that "I don't know" but suggest other questions that can be answered.

Studies :\n\nStudy Link: {context}
Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    
    allm = AzureOpenAI(
    deployment_name=qna_engine, 
    model_name=qna_engine,
    openai_api_key = openai.api_key,
    temperature= .7,
    max_tokens=256)

    chain = LLMChain(llm=allm, prompt=PROMPT, verbose=True)

    #chain = load_qa_chain(allm, chain_type="stuff")

    print(chain.run(context=content, question=search_query+question, return_only_outputs=True))


#add_docs('Covid-19',10)

result = get_documents(20)
print (result)

main()
