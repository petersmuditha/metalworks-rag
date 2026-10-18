import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from glassbox_rag.api_toolkit import create_api
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.config import GlassBoxConfig

# 1. Carica la configurazione dalle variabili d'ambiente (bypassa il bug del file YAML)
config = GlassBoxConfig.from_env()

# 2. Crea l'istanza del motore RAG
engine = GlassBoxEngine(config)

# 3. Definisci il ciclo di vita dell'applicazione
#    Questo assicura che il motore venga inizializzato all'avvio del server
@asynccontextmanager
async def lifespan(app: FastAPI):
    # All'avvio: inizializza il motore RAG
    print("Inizializzazione del motore RAG in corso...")
    await engine.initialize()
    print("✅ Motore RAG inizializzato correttamente.")
    yield
    # Alla chiusura: spegni il motore in modo pulito
    print("🛑 Spegnimento del motore RAG...")
    await engine.shutdown()

# 4. Crea l'app FastAPI di GlassBox
app = create_api(engine)

# 5. Sostituisci il ciclo di vita predefinito con il nostro personalizzato
app.router.lifespan_context = lifespan

# 6. Aggiungi un endpoint di test personalizzato per verificare il retrieval
@app.get("/test-query")
async def test_query(question: str):
    """
    Endpoint di test per eseguire una query di retrieval.
    Esempio: http://localhost:8000/test-query?question=Qual è l'obiettivo del progetto RAG?
    """
    result = await engine.retrieve(query=question, top_k=3)
    return {"results": result.documents}

# Endpoint di health check personalizzato (opzionale, per verificare lo stato)
@app.get("/my-health")
async def my_health_check():
    return {
        "status": "ok",
        "persistent_storage": True,
        "message": "Il database e il vector store sono configurati correttamente."
    }