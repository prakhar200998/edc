from neo4j import GraphDatabase
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embeddings_file = 'graph_embeddings.pkl'
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if not os.path.exists(self.embeddings_file):
            print("Embeddings file not found, generating new embeddings.")
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
 
    def find_similar_entities_with_relationships(self, entity_name):
        if entity_name not in self.entity_embeddings:
            print("Entity not found in current embeddings, encoding now.")
            input_embedding = self.model.encode(entity_name)
        else:
            input_embedding = self.entity_embeddings[entity_name]
        
        # Calculating similarities
        similarities = {}
        for name, embedding in self.entity_embeddings.items():
            similarity = cosine_similarity([input_embedding], [embedding])[0][0]
            similarities[name] = similarity
        
        # Sorting entities based on similarity and selecting the top one
        sorted_entities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        top_entity = sorted_entities[0][0] if sorted_entities else None
        
        if top_entity:
            # Fetching all relationships for the top similar entity
            query = f"MATCH (n)-[r]-(m) WHERE n.name = '{top_entity}' RETURN n.name, r.type as relationship, m.name"
            relationships = self.run_query(query)
            
            # Prepare relationships data for LLM decision-making
            return [{'n.name': record['n.name'], 'relationship': record['relationship'], 'm.name': record['m.name']} for record in relationships]
        else:
            return []

    
    def close(self):
        self.driver.close()




