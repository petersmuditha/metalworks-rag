import os
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Carica le variabili dal file .env (se esiste in locale)
load_dotenv()

# ============================================================
# CONFIGURAZIONE E CREDENZIALI (da variabili d'ambiente)
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY non impostata nell'ambiente.")
if not QDRANT_URL:
    raise RuntimeError("❌ QDRANT_URL non impostata nell'ambiente.")
if not QDRANT_API_KEY:
    raise RuntimeError("❌ QDRANT_API_KEY non impostata nell'ambiente.")

COLLECTION_NAME = "metalworks_defects"
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "models/gemini-embedding-001"


# ============================================================
# INIZIALIZZAZIONE CLIENT
# ============================================================

genai.configure(api_key=GOOGLE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection():
    """Verifica che la collezione esista, altrimenti la crea."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"✅ Collezione '{COLLECTION_NAME}' creata.")
    else:
        print(f"✅ Collezione '{COLLECTION_NAME}' già esistente.")


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Genera l'embedding di un testo con Google Gemini a 1536 dimensioni."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM, # Forza la dimensione corretta
    )
    return result["embedding"]


# ============================================================
# MODELLI PYDANTIC
# ============================================================

class IngestRequest(BaseModel):
    content: str
    filename: str = "documento.txt"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


# ============================================================
# APP FASTAPI
# ============================================================

app = FastAPI(title="MetalWorks Multimodal RAG")


@app.on_event("startup")
async def startup():
    ensure_collection()
    print("✅ Server pronto per ricevere richieste.")


@app.get("/")
def root():
    return {"service": "MetalWorks RAG", "status": "running"}


@app.get("/health")
def health():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values())) if vectors else None
            vector_size = first.size if first else None
        else:
            vector_size = vectors.size
            
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "points": info.points_count,
            "vector_size": vector_size,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Ingerisce un testo, lo chunkka, genera embedding e salva su Qdrant."""
    try:
        # Chunking semplice basato sui paragrafi
        paragraphs = [p.strip() for p in req.content.split("\n\n") if p.strip()]
        chunks = []
        buffer = ""
        for p in paragraphs:
            if len(buffer) + len(p) < 500:
                buffer += ("\n\n" if buffer else "") + p
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = p
        if buffer:
            chunks.append(buffer)

        # Genera embedding e prepara i punti per Qdrant
        points = []
        for chunk in chunks:
            embedding = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"content": chunk, "filename": req.filename},
                )
            )

        # Salva su Qdrant
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {"success": True, "chunks": len(chunks), "filename": req.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest-file")
def ingest_file(filename: str = "miodocumento.txt"):
    """Legge un file dalla cartella data/ e lo ingerisce (comodo per test)."""
    try:
        path = f"data/{filename}"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return ingest(IngestRequest(content=content, filename=filename))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    """Cerca i chunk più simili alla domanda usando Qdrant."""
    try:
        # Genera l'embedding della domanda
        embedding = embed_text(req.question, task_type="RETRIEVAL_QUERY")
        
        # Usa query_points (il nuovo metodo di Qdrant)
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=req.top_k,
        )
        
        return {
            "question": req.question,
            "results": [
                {
                    "content": r.payload.get("content"),
                    "filename": r.payload.get("filename"),
                    "score": r.score,
                }
                for r in response.points
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-query")
def test_query(question: str, top_k: int = 3):
    """Versione GET dell'endpoint di query per test rapidi dal browser."""
    return query(QueryRequest(question=question, top_k=top_k))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

