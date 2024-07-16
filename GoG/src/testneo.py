from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://bb9b1791.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "JjTrdNPlN-QtsB_tHkz-wq9xihhhz7NKl80uGawx5iA"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        result = session.run("RETURN 'Hello, World!' AS message")
        for record in result:
            print(record["message"])
finally:
    driver.close()
