import logging
import azure.functions as func
import logging
import os
import openai
import re
import jmespath
import pandas as pd
import numpy as np
from pytrials.client import ClinicalTrials
from transformers import GPT2Tokenizer
from openai.embeddings_utils import get_embedding, cosine_similarity
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Redis configuration
DIM = 12288
VECT_NUMBER = 3155

# OpenAI Configuration
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

redis_key = os.getenv("REDIS_API_KEY")
redis_endpoint = os.getenv("REDIS_ENDPOINT")

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def get_clinical_study(text , num_studies):
    ct = ClinicalTrials()
    desc = ct.get_full_studies(search_expr=text, max_studies=num_studies)
    detail_lst = []    

    for i in range(num_studies-1):
        detail_lst.append([])
        # Id = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.IdentificationModule.NCTId',desc)
        BriefSummary = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.BriefSummary',desc)
        DetailedDescription = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.DetailedDescription',desc)
        if DetailedDescription is None:
            BriefSummary = normalize_text(BriefSummary)
            detail_lst[i].append(BriefSummary)
        else:
            DetailedDescription = normalize_text(DetailedDescription)
            detail_lst[i].append(DetailedDescription)

    return detail_lst

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def chunk_and_embed(text: str, engine="text-search-davinci-doc-001"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    full_data = {
        "text": text,
        "davinci_search": None
    }

    lenght = len(tokenizer(text)['input_ids'])
    if lenght > 3000:
        return None

    full_data['davinci_search'] = get_embedding(text, engine = 'text-search-davinci-doc-001')

    return full_data

def create_index(redis_conn: Redis, index_name="embeddings-index", prefix = "embedding",number_of_vectors = VECT_NUMBER, distance_metric:str="COSINE"):
    text = TextField(name="text")
    filename = TextField(name="filename")
    embeddings = VectorField("embeddings",
                "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": DIM,
                    "DISTANCE_METRIC": distance_metric,
                    "INITIAL_CAP": number_of_vectors,
                })
    # Create index
    redis_conn.ft(index_name).create_index(
        fields = [text, embeddings, filename],
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    )

def execute_query(np_vector:np.array, return_fields: list=[], search_type: str="KNN", number_of_results: int=20, vector_field_name: str="embeddings"):
    base_query = f'*=>[{search_type} {number_of_results} @{vector_field_name} $vec_param AS vector_score]'
    query = Query(base_query)\
        .sort_by("vector_score")\
        .paging(0, number_of_results)\
        .return_fields(*return_fields)\
        .dialect(2)
    
    params_dict = {"vec_param": np_vector.astype(dtype=np.float32).tobytes()}

    results = redis_conn.ft(index_name).search(query, params_dict)
    return pd.DataFrame(list(map(lambda x: {'id' : x.id, 'text': x.text, 'filename': x.filename, 'vector_score': x.vector_score}, results.docs)))

def get_documents(number_of_results: int=VECT_NUMBER):
    base_query = f'*'
    return_fields = ['id','text','filename']
    query = Query(base_query)\
        .paging(0, number_of_results)\
        .return_fields(*return_fields)\
        .dialect(2)
    results = redis_conn.ft(index_name).search(query)
    if results.docs:
        return pd.DataFrame(list(map(lambda x: {'id' : x.id, 'text': x.text, 'filename': x.filename}, results.docs))).sort_values(by='id')
    else:
        return pd.DataFrame()

def set_document(elem):
    index = str(uuid.uuid4())
    redis_conn.hset(
        f"embedding:{index}",
        mapping={
            "text": elem['text'],
            "filename": elem['filename'],
            "embeddings": np.array(elem['davinci_search']).astype(dtype=np.float32).tobytes()
        }
    )

def delete_document(index):
    redis_conn.delete(f"{index}")

# Connect to the Redis server
redis_conn = Redis(host= os.environ.get('REDIS_ADDRESS','localhost'), port=6379, password=os.environ.get('REDIS_PASSWORD',None)) #api for Docker localhost for local execution

# Check if Redis index exists
index_name = "embeddings-index"
try:
    if redis_conn.ft(index_name).info():
        print("Index exists")
except:
    print("Index does not exist")
    print("Creating index")
    # Create index 
    create_index(redis_conn)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
