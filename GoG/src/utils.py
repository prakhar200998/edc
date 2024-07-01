import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_entities(query):
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

def create_prompt_from_subgraph(subgraph_data):
    context = " ".join([f"{data['n_name']} is connected to {data['m_name']} through {data['relationship']}" for data in subgraph_data])
    return context
