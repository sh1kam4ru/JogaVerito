from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI()
BASE_DIR = Path(__file__).parent.resolve()
print(BASE_DIR)
templates = Jinja2Templates(directory=Path(BASE_DIR, 'static'))
app.mount("/static", StaticFiles(directory=Path(BASE_DIR, 'static')), name="static")

google_api_key = os.environ.get('GOOGLE_API_KEY')
qdrant_url = os.environ.get('QDRANT_URL')
qdrant_api_key = os.environ.get('QDRANT_API_KEY')
collection = "EdenHazard"

def build_context(query):
    global model
    global qdrant_url
    global qdrant_api_key
    model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = model.embed_query(query)
    client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
    )
    hits = client.search(collection_name=collection, query_vector=embeddings, limit=5)
    threshold_score = 0.15
    context = ""
    all_sources = []
    for i in range(len(hits)):
        hit = hits[i]
        print(hit.score * hit.score)
        context += f"ID: {i+1} \nInformation: {hit.payload['stat']} \n\n"
        all_sources.append(hit.payload["source"])

    return context, all_sources

def partition_string(input_string):
    print(input_string)
    sentences = input_string.split("^^")
    ids_sentence = sentences[-1]
    answer = '. '.join(sentences[:-1])

    return answer, ids_sentence

def extract_numbers(input_string):
    numbers = re.findall(r'\b\d+\b', input_string)
    return numbers

def extract_sources(ids_used, all_sources):
    sources_used = []
    for i in range(len(ids_used)):
        id = int(ids_used[i])
        sources_used.append(all_sources[id-1])

    return sources_used

async def query_checker(query):
    global google_api_key
    try:
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        prompt_template = """
        You are football Q&A bot. You have expert knowledge regarding Eden Hazard (often referred to as Eden or Hazard or Azza or Azzar) and his footballing career. You DO NOT know anything else.
        Consider the following question: {query}. 
        If the question is regarding Eden Hazard or Hazard AND his football career, say 'yes', otherwise say 'no'. You must strictly say just 'yes' or 'no' and nothing else.
        """
        prompt = PromptTemplate(
            input_variables=["query"], template=prompt_template
        )
        llmchain = LLMChain(llm=llm, prompt=prompt)
        result = llmchain.run({"query": query})
        return result
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

async def llm_ans(query):
    global google_api_key
    try:
        context, all_sources = build_context(query)
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        prompt_template = "{context} \nUsing the information given above, answer the following question: {query}. Construct a crisp and complete answer using not more than 100 words. Use a fullstop(.) after each sentence. Don't say something that is not related to the question. Strictly print only the ID numbers you have used to answer the question at the END of your answer by filling in this template: ^^IDs = . Do not reference the ID numbers in your answer."
        prompt = PromptTemplate(
            input_variables=["context", "query"], template=prompt_template
        )
        llmchain = LLMChain(llm=llm, prompt=prompt)
        result = llmchain.run({"context": context, "query": query})
        answer, ids_sentence = partition_string(result)
        ids_used = extract_numbers(ids_sentence)
        sources_used = extract_sources(ids_used, all_sources)
        return answer, sources_used
    except Exception as e:
        raise Exception(f"Error: {str(e)}")
    
@app.get("/health")
def home():
    return {"health_check": "OK"}

@app.get("/")
def read_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
async def wizard(question: str = Query(..., title="Your Question", description="Enter your question")):
    print(question)
    result = await query_checker(question)
    if "yes".lower() in result.lower():
        ans, sources = await llm_ans(question)
    else:
        ans = "I'm sorry, I do not know the answer to this question."
        sources = []
    return {"answer": ans, "sources": sources}
