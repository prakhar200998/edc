from neo4j import GraphDatabase
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embeddings_file = 'graph_embeddings.pkl'
        if not os.path.exists(self.embeddings_file):
            print("Embeddings file not found, generating new embeddings.")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.fetch_and_encode_entities()
        else:
            print("Loading embeddings from file.")
            self.entity_embeddings = self.load_embeddings()
            print(f"Loaded {len(self.entity_embeddings)} embeddings.")

    def run_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    def fetch_and_encode_entities(self):
        print("Fetching entities from Neo4j...")
        query = "MATCH (n) RETURN n.name AS name"
        results = self.run_query(query)
        entity_names = [result['name'] for result in results]
        print(f"Fetched {len(entity_names)} entities.")
        print("Encoding entities...")
        self.entity_embeddings = {name: self.model.encode(name) for name in entity_names}
        print("Entities encoded. Saving embeddings...")
        self.save_embeddings()

    def save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.entity_embeddings, f)
        print("Embeddings saved to file.")

    def load_embeddings(self):
        with open(self.embeddings_file, 'rb') as f:
            return pickle.load(f)

    def print_sample_embeddings(self):
        print("Sample Embeddings:")
        for name, embedding in list(self.entity_embeddings.items())[:5]:
            print(f"{name}: {embedding}")
    
    def close(self):
        self.driver.close()
