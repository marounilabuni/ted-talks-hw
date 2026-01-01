import pandas as pd
import json
import pickle
import os
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec


def row_to_chunks(
    row: pd.Series,
    chunk_size: int = 1024,        # approx tokens
    overlap: int = 150            # approx tokens (≤ 30%)
) -> List[Dict]:
    """
    Convert a single TED Talk row into overlapping transcript chunks.

    Returns a list of dicts, each representing one chunk with metadata.
    """

    #transcript = row["transcript"]
    transcript = row["description"]

    # Handle missing / empty transcripts safely
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Transcript is not a string or is empty")

    # Simple token approximation: split by whitespace
    tokens = transcript.split()

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        chunk_text = " ".join(chunk_tokens)

        chunks.append({
            "talk_index":row.name,
            "talk_id": row["talk_id"],
            "title": row["title"],
            "speaker": row["speaker_1"],
            "chunk_index": chunk_index,
            "chunk_text": chunk_text
        })

        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def get_all_chunks(df: pd.DataFrame):
    all_chunks = []
    chunks_counts_pre_doc = []
    for _, row in df.iterrows():
        _chunks = row_to_chunks(row,)
        all_chunks.extend(_chunks)
        chunks_counts_pre_doc.append(len(_chunks))

    return all_chunks, chunks_counts_pre_doc



def embed_texts(texts: List[str], model_name: str = "text-embedding-3-small", embeddings_model=None) -> Tuple[list[list[float]], OpenAIEmbeddings]:
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings(model=model_name)
    else:
        embeddings_model = embeddings_model
        
    lengths = [len(text) for text in texts]
    chunks = []
    LIMIT = 300_000 # limit of tokens per request
    index = 0
    while index < len(texts):
        chunk = []
        for d in range(len(texts) - index):
            if sum(lengths[index:index+d]) < LIMIT:
                chunk.append(texts[index+d])
            else:
                break
        chunks.append(chunk)
        index += len(chunk)
    
    embeddings = [embeddings_model.embed_documents(chunk) for chunk in tqdm(chunks)]
    embeddings = sum(embeddings, [])

    return embeddings, embeddings_model



def search_similar(
    query: str,
    model,
    embeddings_db,
    k: int = 5
):
    
    q_emb = model.embed_documents([query])#.astype("float32")    
    
    res = embeddings_db.query(
        vector=q_emb.tolist(),
        top_k=k,
        include_metadata=True
    )

    results = []
    for match in res["matches"]:
        results.append((match["score"], match["metadata"]))
    
    return results


# build the database
def build_pinecone_index(pc, index_name: str, df):
    pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
    
    index = pc.Index(index_name)

    all_chunks, chunks_counts_pre_doc = get_all_chunks(df)
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings, model = embed_texts(texts)
    vectors = []
    for i in range(len(embeddings)):
        vector = (
            f"chunk-{i}",
            embeddings[i],
            all_chunks[i]
        )
        vectors.append(vector)
        if len(vectors) == 100 or i == len(embeddings) - 1:
            index.upsert(vectors)
            vectors.clear()
    
    return index



# initialize the pinecone index
def init_pinecone(df=None):
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
    )
    
    INDEX_NAME = "ted-talks"
    if False:
        try:
            pc.delete_index(INDEX_NAME)
        except: pass
    
    index_list = pc.list_indexes()
    index_names_list = [i["name"] for i in index_list]
    if INDEX_NAME not in index_names_list:
        print(f"Building new index: {INDEX_NAME}")
        if df is None:
            raise ValueError("df is required to build the index")
        return build_pinecone_index(pc, INDEX_NAME, df=df)

    return pc.Index(INDEX_NAME)
