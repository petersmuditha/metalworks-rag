import asyncio
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.config import load_config

async def main():
    # Carica la configurazione (che ora legge correttamente le tue variabili d'ambiente)
    config = load_config("config/default.yaml")
    
    # Inizializza il motore RAG
    engine = GlassBoxEngine(config)
    await engine.initialize()
    
    # Leggi il documento e prepara i dati per l'ingestione
    with open("data/miodocumento.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    document = {"content": content, "metadata": {"filename": "miodocumento.txt"}}
    
    print("Inizio ingestione...")
    # Chiama direttamente il metodo di ingestione del motore
    result = await engine.ingest(documents=[document])
    print("Ingestione completata!", result)
    
    # Chiudi il motore
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())