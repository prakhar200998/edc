import ast
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URL = "neo4j+s://beaf91c7.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "OfOkycCG98zby9IztPN-Bb_k6ycyF73N3NDKSDKZRV4"

def sanitize_relationship(relationship):
    # Replace invalid characters with underscores and convert to uppercase
    return relationship.replace(" ", "_").replace("-", "_").upper()

def read_triples(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    line_triples = ast.literal_eval(line)
                    if isinstance(line_triples, list):
                        triples.extend(line_triples)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing line: {line}. Error: {e}")
    return triples

def create_graph(driver, triples):
    with driver.session() as session:
        for triple in triples:
            if len(triple) == 3:
                session.execute_write(create_relationship, triple)

def create_relationship(tx, triple):
    subject, predicate, obj = triple
    sanitized_predicate = sanitize_relationship(predicate)
    query = (
        "MERGE (s:Entity {name: $subject}) "
        "MERGE (o:Entity {name: $object}) "
        "MERGE (s)-[r:" + sanitized_predicate + "]->(o)"
    )
    tx.run(query, subject=subject, object=obj)

def main():
    file_path = '../output/chemotherapy_schema_canonicalization/edc_output.txt'
    triples = read_triples(file_path)
    
    if not triples:
        print("No valid triples found. Exiting.")
        return
    
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Delete existing nodes and relationships in the database
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    create_graph(driver, triples)
    driver.close()

if __name__ == "__main__":
    main()
