from qdrant_client import QdrantClient

# Incolla qui i tuoi dati
QDRANT_URL = "https://a327d02b-fb0a-4762-a220-5bd9026c368e.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NDNkNGUxNjgtZWU2NS00NGUzLWFjZTYtNjZlMjg5NzBmYTBmIn0.B0WV0hfKpFIqv96Jc5l0zdQ9iKkwERbLSrEDHrvKwYY"

print("Tentativo di connessione a Qdrant...")
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections()
    print("✅ Connessione riuscita!")
    print("Collezioni trovate:", collections)
except Exception as e:
    print("❌ Errore di connessione:")
    print(e)