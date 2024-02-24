from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
google_api_key = os.environ.get('GEMINI_API_KEY')
qdrant_url = os.environ.get('QDRANT_URL')
qdrant_api_key = os.environ.get('QDRANT_API_KEY')
collection = "EdenHazard"

def build_context(query):
    instruction = "Represents a football information: "
    embeddings = model.encode([instruction+query], normalize_embeddings=True)
    client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
    )
    hits = client.search(collection_name=collection, query_vector=embeddings[0], limit=5)
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

async def llm_ans(query):
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

@app.get("/")
def read_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
async def wizard(question: str = Query(..., title="Your Question", description="Enter your question")):
    ans, sources = await llm_ans(question)
    return {"answer": ans, "sources": sources}
