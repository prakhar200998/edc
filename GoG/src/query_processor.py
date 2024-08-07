from neo4j_connection import Neo4jConnection
from utils import extract_entities, extract_relations, find_similar_entities, fetch_subgraph_for_entities
from neo4j_connection import Neo4jConnection

def handle_query(user_query):
    neo4j_conn = Neo4jConnection()

    try:
        
        #neo4j_conn.print_sample_embeddings()
        
        entities = extract_entities(user_query)
        relations = extract_relations(user_query)
        print(f"Extracted Entities: {entities}")
        print(f"Extracted Relations: {relations}")

        similarities = find_similar_entities(entities, neo4j_conn.entity_embeddings)
        print(f"Similar Entities: {similarities}")

        # subgraph = fetch_subgraph_for_entities(neo4j_conn, similarities, relations)
        # print("Extracted Subgraph:")
        # for record in subgraph:
        #     print(f"{record['n']['name']} -[{record['r']['type']}]-> {record['m']['name']}")

       
    
        
        similar_entities = neo4j_conn.find_similar_entities_with_relationships("Chemotherapy")
        for result in similar_entities:
            print(f"Entity: {result['entity']} with similarity: {result['similarity']}")
            print("Relationships:")
            for rel in result['relationships']:
                print(f"{rel['n.name']} - {rel['type(r)']} - {rel['m.name']}")

         # For now, return a placeholder response
        return "Subgraph extraction completed. See printed results above."
        
    finally:
        neo4j_conn.close()
