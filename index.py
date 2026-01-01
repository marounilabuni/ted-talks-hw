#import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec

from dotenv import load_dotenv

from utils import init_pinecone, search_similar

from flask import Flask, jsonify, request

from constants import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from constants import SYSTEM_PROMPT, user_prompt_template

def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


load_dotenv()



# load the data
#import pandas as pd
#df = pd.read_csv("ted_talks_en.csv")
df = None
embeddings_db = init_pinecone(df)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(
    model="RPRTHPB-gpt-5-mini",
    api_key=os.environ["LLMOD_API_KEY"],
    base_url="https://api.llmod.ai",
)

# to enforce the output format (no extra text, only the required format)
class FinalResponse(BaseModel):
    final_response: str = Field(description="""
    The natural textual response to give the user
    """)
    # additional fields if needed...

structured_llm = llm.with_structured_output(FinalResponse)


def run_pipeline(question: str, df):
    similar_records = search_similar(question, embeddings_model, embeddings_db, k = 20)
    
    returned_chunks = [c for i, c in similar_records]
    
    textual_context = ""
    aggrigated_chunks = {}
    for i in range(len(returned_chunks)):
        #returned_chunks[i]['score'] = similar_records[i][1]
        if returned_chunks[i]['talk_id'] not in aggrigated_chunks:
            aggrigated_chunks[returned_chunks[i]['talk_id']] = []
        aggrigated_chunks[returned_chunks[i]['talk_id']].append(returned_chunks[i])
    
    for talk_id, chunks in aggrigated_chunks.items():
        speaker = 'Unknown'
        if 'speaker' in chunks[0]:
            speaker = chunks[0]['speaker']
        else:
            try:
                talk_id = chunks[0]['talk_id']
                speaker = df[df['talk_id'] == talk_id]['speaker_1'].values[0]
            except:
                pass
        textual_context += f"\n---\nTitle: {chunks[0]['title']}"
        textual_context += f"\nSpeaker: {speaker}"
        textual_context += f"\nRelative context:"
        for chunk in chunks:
            textual_context += f"\n\t{chunk['chunk_text']}"
        
        
    prompt = user_prompt_template.format(textual_context=textual_context, question=question)

    p = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]
    
    
    
    response = structured_llm.invoke(p)
    return {
        "response": response.final_response,
        "context": returned_chunks,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": prompt
        }
    }
    


# Flask app

app = Flask(__name__)

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())

@app.route("/api/prompt", methods=["POST"])
def prompt():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    
    if not isinstance(data['question'], str):
        return jsonify({"error": "Question must be a string"}), 400
    
    question = data['question']
    
    try:
        resp = run_pipeline(question, df)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)