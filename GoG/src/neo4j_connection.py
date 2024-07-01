from neo4j import GraphDatabase
import pickle
import os

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.embeddings_file = 'graph_embeddings.pkl'
        self.entity_embeddings = self.load_embeddings()

    def close(self):
        self.driver.close()

    def run_query(self, query):
        with self.driver.session() as session:
            return [record for record in session.run(query)]

    def fetch_and_encode_entities(self, model):
        query = "MATCH (n) RETURN n.name AS name"
        results = self.run_query(query)
        entity_names = [result['name'] for result in results]
        embeddings = model.encode(entity_names, convert_to_tensor=True)
        self.entity_embeddings = dict(zip(entity_names, embeddings))
        self.save_embeddings()

    def save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.entity_embeddings, f)

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                return pickle.load(f)
        return {}
