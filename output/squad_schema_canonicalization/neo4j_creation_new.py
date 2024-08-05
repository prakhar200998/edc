import re
from neo4j import GraphDatabase

# Database credentials and connection URI
uri = "neo4j+s://bb9b1791.databases.neo4j.io"
username = "neo4j"
password = "JjTrdNPlN-QtsB_tHkz-wq9xihhhz7NKl80uGawx5iA"

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.write_transaction(lambda tx: tx.run(query, parameters))
            return result

def sanitize(value):
    # Replace special characters with underscores, preserve decimal points
    sanitized = re.sub(r'[^0-9a-zA-Z_.]', '_', value)
    sanitized = re.sub(r'\.', '__', sanitized)  # Replace decimal points with double underscores
    sanitized = re.sub(r'^[^a-zA-Z]+', '', sanitized)
    return sanitized

def process_file(filename, neo4j_conn):
    with open(filename, 'r') as file:
        content = file.read().strip()

    # Split the content into groups and triples manually
    groups = content.split('],[')
    for group in groups:
        # Remove extra brackets
        group = group.strip('[]')
        # Split the group into triples
        triples = group.split('], [')
        for triple in triples:
            # Remove extra quotes and spaces, then split by ', '
            triple = triple.strip('[]').replace("'", "").split(', ')
            if len(triple) == 3:
                subject, predicate, obj = triple
                sanitized_subject = sanitize(subject)
                sanitized_predicate = sanitize(predicate)
                sanitized_obj = sanitize(obj)
                
                # Create a Cypher query to create the nodes and relationships using parameters
                cypher_query = f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                MERGE (a)-[r:{sanitized_predicate}]->(b)
                """
                parameters = {
                    'subject': sanitized_subject,
                    'object': sanitized_obj
                }
                
                # Print statements for debugging
                print(f"Executing query with parameters: {parameters}")
                
                # Execute the query
                neo4j_conn.execute_query(cypher_query, parameters)

# Connect to Neo4j
conn = Neo4jConnection(uri, username, password)

# Clear existing graph
print("Clearing existing graph...")
conn.execute_query("MATCH (n) DETACH DELETE n")

file_to_process = 'edc_output.txt'
# Process the file and create the graph
print(f"Processing file: {file_to_process}")
process_file(file_to_process, conn)

# Close the connection
conn.close()
print("Connection closed.")
