from sentence_transformers import SentenceTransformer, util
import spacy
import torch
from openai import OpenAI
from config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def extract_relations(query):
    prompt = f"""
    Extract the most relevant relation from the following query in the context of chemotherapy-related information:
    
    Query: "What are the side effects of chemotherapy?"
    Relation: "HAS_SIDE_EFFECTS"
    
    Query: "What drugs are used in chemotherapy?"
    Relation: "USES_DRUGS"
    
    Query: "How does chemotherapy affect the immune system?"
    Relation: "AFFECTS_IMMUNE_SYSTEM"
    
    Query: "{query}"
    Relation:
    """

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )

    relation = completion.choices[0].message.content
    return relation



def find_similar_entities(query_entities, entity_embeddings, threshold=0.8, top_n=5):
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Encoding query entities
    query_embeddings = model.encode(query_entities, convert_to_tensor=True).to(device)
    
    # Dictionary to store similarities
    similarities = {}
    
    # Calculate cosine similarities
    for name, embedding in entity_embeddings.items():
        embedding = torch.tensor(embedding).to(device) if not isinstance(embedding, torch.Tensor) else embedding.to(device)
        similarity = util.cos_sim(query_embeddings, embedding.unsqueeze(0))
        similarities[name] = similarity
    
    # Filter based on the threshold and sort by similarity value
    filtered_similarities = {k: v for k, v in similarities.items() if v.item() > threshold}
    sorted_similarities = sorted(filtered_similarities.items(), key=lambda item: item[1], reverse=True)
    similarities = {k: v.item() for k, v in sorted_similarities[:top_n]}
    # Return top N entities
    return similarities



def fetch_subgraph_for_entities(neo4j_conn, matched_entities, extracted_relation):
    subgraph_data = []
    entity_names = list(matched_entities.keys())

    for entity in entity_names:
        query = f"""
        MATCH (n {{name: '{entity}'}})-[r*1..2]-(m)
        WHERE any(rel in r WHERE type(rel) = '{extracted_relation}')
        RETURN n, r, m
        """
        results = neo4j_conn.run_query(query)
        if results:
            for result in results:
                try:
                    # Safely access elements of result if not None
                    n_name = result['n']['name'] if 'n' in result and result['n'] else "Unknown Node"
                    m_name = result['m']['name'] if 'm' in result and result['m'] else "Unknown Node"
                    if 'r' in result and result['r']:
                        relation_type = ', '.join([rel['type'] for rel in result['r']])
                    else:
                        relation_type = "Unknown Relation"
                    record = f"{n_name} -[{relation_type}]-> {m_name}"
                    subgraph_data.append(record)
                except KeyError as e:
                    print(f"Error accessing result data: {e}")
                except IndexError as e:
                    print(f"Index error: {e}")
        else:
            print(f"No results for entity: {entity}")
    return subgraph_data





