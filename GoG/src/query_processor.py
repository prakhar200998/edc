from neo4j_connection import Neo4jConnection
from openai_connection import OpenAIConnection
from utils import extract_entities, find_similar_entities, fetch_subgraph_for_entities

def handle_query(user_query):
    neo4j_conn = Neo4jConnection('uri', 'username', 'password')
    gpt_conn = OpenAIConnection('api_key')
    entities = extract_entities(user_query)
    similarities = find_similar_entities(entities, neo4j_conn.entity_embeddings)
    subgraph = fetch_subgraph_for_entities(neo4j_conn, similarities)
    # Use subgraph and GPT to process query further
    # Detailed logic here
    neo4j_conn.close()
