from pytrials.client import ClinicalTrials
import jmespath
import re
from openai.embeddings_utils import get_embedding, cosine_similarity
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import GPT2Tokenizer
from langchain.llms import AzureOpenAI
import openai
import uuid
import numpy as np
import pandas as pd
import streamlit as st

openai.api_type = "azure"
openai.api_base = 'https://endpoint.openai.azure.com/'
openai.api_version = "2022-12-01"
openai.api_key = '2ea87dbe6c0140fc8a7dd4446a8243c6'
redis_key = '2Ek0dViRcVV77ERAsZET+kuzhGZkRdkkKIKaIYhzbJ4='
redis_endpoint = 'redissearch.southcentralus.redisenterprise.cache.azure.net'

# Redis configuration
#Ada: 1024 dimensions
#Babbage: 2048 dimensions
#Curie: 4096 dimensions
#Davinci: 12288 dimensions
DIM = 1024
VECT_NUMBER = 3155
index_name = "embeddings-index"
url_prefix = "https://clinicaltrials.gov/ct2/show/"

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

def get_clinical_study(text , num_studies):
    ct = ClinicalTrials()
    desc = ct.get_full_studies(search_expr=text, max_studies=num_studies)
    detail_lst = []    

    for i in range(num_studies):
        detail_lst.append([])
        NCTId = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.IdentificationModule.NCTId',desc)
        source_url = url_prefix + NCTId
        #RecruitmentStatus = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.StatusModule.OverallStatus',desc)
        #print (RecruitmentStatus)

        BriefSummary = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.BriefSummary',desc)
        DetailedDescription = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.DescriptionModule.DetailedDescription',desc)
        EligibilityCriteria = jmespath.search('FullStudiesResponse.FullStudies['+str(i)+'].Study.ProtocolSection.EligibilityModule.EligibilityCriteria',desc)
        
        EligibilityCriteria = normalize_text(EligibilityCriteria)
        if DetailedDescription is None:
            BriefSummary = normalize_text(BriefSummary)
            detail_lst[i].append(source_url)
            detail_lst[i].append(BriefSummary + EligibilityCriteria)
        else:
            DetailedDescription = normalize_text(DetailedDescription)
            detail_lst[i].append(source_url)
            detail_lst[i].append(BriefSummary + DetailedDescription + EligibilityCriteria)

    return detail_lst

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def chunk_and_embed(source_url, text: str, engine="text-search-ada-doc-001"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    full_data = {
        "source_url" : source_url,
        "text": text,
        "ada_search": None
    }

    lenght = len(tokenizer(text)['input_ids'])
    if lenght > 2000:
        return None

    full_data['ada_search'] = get_embedding(text, engine = 'text-search-ada-doc-001')

    return full_data

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
            "embeddings": np.array(elem['ada_search']).astype(dtype=np.float32).tobytes()
        }
    )

def add_embeddings(source_url, text):
    embeddings = chunk_and_embed(source_url, text)
    if embeddings:
        # Store embeddings in Redis
        set_document(embeddings)
    else:
        st.error("No embeddings were created for this document as it's too long. Please keep it under 3000 tokens")

def get_documents(number_of_results: int=VECT_NUMBER):
    base_query = f'*'
    return_fields = ['id', 'source_url', 'text']
    query = Query(base_query)\
        .paging(0, number_of_results)\
        .return_fields(*return_fields)\
        .dialect(2)
    results = redis_conn.ft(index_name).search(query)
    if results.docs:
        return pd.DataFrame(list(map(lambda x: {'id' : x.id, 'source_url': x.source_url, 'text': x.text}, results.docs))).sort_values(by='id')
    else:
        return pd.DataFrame()


def execute_query(np_vector:np.array, return_fields: list=[], search_type: str="KNN", number_of_results: int=20, vector_field_name: str="embeddings"):
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
def search_semantic_redis(search_query, n=3, pprint=True, engine='ada'):
    embedding = get_embedding(search_query, engine= 'text-search-{engine}-query-001')
    res = execute_query(np.array(embedding))

    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res.reset_index()

# Return a semantically aware response using the Completion endpoint
def get_semantic_answer(df, question, explicit_prompt="", model="DaVinci-text", engine='babbage', limit_response=True, tokens_response=100, temperature=0.7):

    restart_sequence = "\n\n"
    question += "\n"

    res = search_semantic_redis(df, question, n=3, pprint=False, engine=engine)

    if len(res) == 0:
        prompt = f"{question}"
    elif limit_response:
        res_text = "\n".join(res['text'][0:int(os.getenv("NUMBER_OF_EMBEDDINGS_FOR_QNA",1))])
        question_prompt = explicit_prompt.replace(r'\n', '\n')
        question_prompt = question_prompt.replace("_QUESTION_", question)
        prompt = f"{res_text}{restart_sequence}{question_prompt}"
    else:
        prompt = f"{res_text}{restart_sequence}{question}"
            

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=tokens_response,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    print(f"{response['choices'][0]['text'].encode().decode()}\n\n\n")

    return prompt,response#, res['page'][0]

def delete_index(index):
    result = redis_conn.execute_command("FT.DROP", index)  
    print('Index Deleted.')

def create_context(question):
    """
    Find most relevant context for a question via Redisearch 
    """
    q_embed = search_semantic_redis(question, engine=f'text-search-{size}-query-001')
    res = index.query(q_embed, top_k=5, include_metadata=True)
    

    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts)


def main(study_name, number_of_studies):
    # Check if Redis index exists
    index_name = "embeddings-index"
    delete_index(index_name)
    try:
        if redis_conn.ft(index_name).info():
            print("Index exists")
    except:
        print("Index does not exist")
        print("Creating index")
        # Create index 
        create_index(redis_conn)

    studis = get_clinical_study(study_name, number_of_studies)
    #print (studis)
    for inner_list in studis:
        #print (inner_list[0])
        add_embeddings(inner_list[0],inner_list[1])
    result = get_documents(number_of_studies)
    return result

print(main('Coronavirus+COVID',1 ))

llm = AzureOpenAI(
    deployment_name="text-davinci-002", 
    model_name="text-davinci-002",
    openai_api_key = openai.api_key
)

print(llm("Tell me a joke"))




