from neo4j import GraphDatabase

# Database credentials and connection URI
uri = "neo4j+s://beaf91c7.databases.neo4j.io"
username = "neo4j"
password = "OfOkycCG98zby9IztPN-Bb_k6ycyF73N3NDKSDKZRV4"

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.write_transaction(lambda tx: tx.run(query, parameters))
            return result

def process_file(filename, neo4j_conn):
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            subject = ' '.join(parts[0:-2])  # Everything except the last two words
            relationship = parts[-2]
            object = parts[-1]
            
            # Create a Cypher query to create the nodes and relationships using parameters
            cypher_query = """
            MERGE (a:Entity {name: $subject})
            MERGE (b:Entity {name: $object})
            MERGE (a)-[r:RELATIONSHIP {type: $relationship}]->(b)
            """
            parameters = {
                'subject': subject,
                'relationship': relationship,
                'object': object
            }
            
            # Execute the query
            neo4j_conn.execute_query(cypher_query, parameters)

# Connect to Neo4j
conn = Neo4jConnection(uri, username, password)

# Process the file and create the graph
process_file('edc_output_processed.txt', conn)

# Close the connection
conn.close()
