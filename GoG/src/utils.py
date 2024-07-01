from sentence_transformers import SentenceTransformer, util
import spacy

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def find_similar_entities(query_entities, entity_embeddings):
    query_embeddings = model.encode(query_entities, convert_to_tensor=True)
    similarities = {}
    for name, embedding in entity_embeddings.items():
        similarity = util.cos_sim(query_embeddings, embedding.unsqueeze(0))
        similarities[name] = similarity
    return similarities

def fetch_subgraph_for_entities(neo4j_conn, matched_entities):
    subgraph_data = []
    for entity in matched_entities:
        query = f"MATCH (n {{name: '{entity}'}})-[*1..2]-(m) RETURN n, m"
        results = neo4j_conn.run_query(query)
        subgraph_data.extend(results)
    return subgraph_data
