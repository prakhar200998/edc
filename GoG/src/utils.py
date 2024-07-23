from sentence_transformers import SentenceTransformer, util
import spacy
import torch
from openai import OpenAI
from config import OPENAI_API_KEY
from neo4j_connection import Neo4jConnection
import re


client = OpenAI(api_key=OPENAI_API_KEY)
neo4j_conn = Neo4jConnection()
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )

    relation = completion.choices[0].message.content
    return relation


import openai

def extract_response(question, instruction, example):
    """
    Generates a response for a given question using interleaved Thought, Action, Observation steps.

    :param question: The user's question to answer.
    :param instruction: A string of instructions on how to process the question.
    :param example: A string of example flow demonstrating how to process a question.
    :return: The response from the LLM.
    """
    # Combining instruction, example, and the actual question to form the full prompt
    prompt = f"{instruction}\n{example}\nQuestion: {question}\n"

    try:
        # Using OpenAI's API to get a response based on the prompt
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,  # Adjust max_tokens as needed to allow the model to generate full responses
            temperature=0.5   # A balanced temperature for creative yet controlled outputs
        )

        # Extracting and returning the response content
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        # Error handling
        print(f"An error occurred: {e}")
        return None
    
    
# def extract_response_new(question, instruction, example):
#     """
#     Generates a response for a given question using interleaved Thought, Action, Observation steps.
#     Runs up to 6 times or until a 'Finish' action is reached.

#     :param client: OpenAI client instance for API access.
#     :param question: The user's question to answer.
#     :param instruction: A string of instructions on how to process the question.
#     :param example: A string of example flow demonstrating how to process a question.
#     :param neo4j_conn: Instance of Neo4jConnection for database interactions.
#     :return: The final response from the LLM.
#     """
#     # Combining instruction, example, and the actual question to form the full prompt
#     prompt = f"{instruction}\n{example}\nQuestion: {question}\n"

#     action = ""
#     i = 0
#     while not re.search(r"\bFinish\b", action):
#         i += 1
#         try:
#             print(f"Round {i} - Generating response with prompt")
#             # Using OpenAI's API to get a response based on the prompt
#             completion = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=1024,  # Adjust max_tokens as needed
#                 temperature=0.5   # Balanced temperature for controlled outputs
#             )

#             # Extracting the response content
#             response = completion.choices[0].message.content
#             print(f"{response.strip()}")  # Print the response per round

#             # Parse the response to find the next action
#             thought, action = parse_response(response)
#             print(f"Thought: {thought}\n Action: {action}")

#             if "Search" in action:
#                 print(f"Entered search phase")
#                 entity = action.split("[")[1].rstrip("]")
#                 print(f"Executing search for entity: {entity}")
#                 search_results = neo4j_conn.find_similar_entities_with_relationships(entity)
#                 print(f"Search Results for {entity}: {search_results}")
#                 top_relationships = select_top_relationships(search_results, thought)
#                 print(f"Top Relationships: {top_relationships}")
#                 observation = "; ".join([f"{rel['n.name']} - {rel['relationship']} - {rel['m.name']}" for rel in top_relationships])
#                 prompt += f"\nObservation {i}: {observation}"
#             elif "Generate" in action:
#                 prompt += f"\nObservation {i}: Generated new triples based on current knowledge."
#             elif "Finish" in action:
#                 print("Finish action encountered. Terminating the process.")
#                 break

#             # prompt += f"\nThought {i+1}: {thought}\nAction {i+1}: {action}"
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             break

#     return response.strip()

def generate_new_triple(thought, client):
    """
    Simulates generating a new triple based on the thought and the inherent knowledge of the LLM.
    """
    # Example prompt to generate a triple based on the thought
    prompt = f"Based on the following thought: '{thought}' and the observations previously fed in the prompt, generate a relevant triple."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate a triple."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.5
    )
    return completion.choices[0].message.content.strip()

def flatten_observations(observations):
    """Flatten a list of lists into a single list of strings."""
    flat_list = []
    for sublist in observations:
        if isinstance(sublist, list):
            # Extend flat_list with elements from sublist if they are not lists themselves
            flat_list.extend([str(item) for item in sublist if not isinstance(item, list)])
        else:
            flat_list.append(str(sublist))
    return flat_list


def add_triple_to_graph(neo4j_conn, triple):
    """
    Adds the generated triple to the Neo4j graph after validating and resolving entities.
    """
    subj, rel, obj = triple.split(", ")
    # Here you'd include logic to handle entity matching, creation, and relationship linking.
    query = f"MERGE (a:Entity {{name: '{subj}'}}) MERGE (b:Entity {{name: '{obj}'}}) MERGE (a)-[:{rel}]->(b)"
    neo4j_conn.run_query(query)
    return f"Added to graph: {triple}"

def generate_final_answer(answer_list, observations):
    """
    Uses an LLM to generate a comprehensive answer based on provided answers and observations.

    :param answer_list: A list of potential answers or key information snippets.
    :param observations: A list of observations made during the interaction.
    :return: A synthesized answer as a string.
    """
    # Convert list to formatted string
    answers = ', '.join(answer_list).strip()
    
    # Flatten and create a summary of observations
    flat_observations = flatten_observations(observations)
    observation_summary = ' '.join(flat_observations)
    
    # Construct the prompt for the LLM
    prompt = f"Based on the following observations: {observation_summary}, and considering these potential answers: {answers}, please synthesize a concise and comprehensive answer to the question."

    # Call to the LLM API
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI trained to integrate information into clear responses."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,  # Adjust as necessary
        temperature=0.5  # Keeping it more deterministic
    )

    # Retrieve and return the LLM-generated response
    final_answer = completion.choices[0].message.content.strip()
    return final_answer

def extract_response_new(question, instruction, example):
    """
    Generates a response for a given question using interleaved Thought, Action, Observation steps.
    The LLM generates a Thought and an Action based initially on the question, and subsequently based on added observations.

    :param question: The user's question to answer.
    :param instruction: Instructions on how to process the question.
    :param example: Example flow demonstrating how to process a similar question.
    :return: The final response from the LLM after completing the task or reaching the maximum number of rounds.
    """
    # Start the initial prompt with instruction, example, and user question
    prompt = f"{instruction}\n{example}\nQuestion: {question}\n"
    observation_list = []
    # Variables to control the loop
    i = 0
    action = ""
    print(f"Question: {question}")
    while i < 6 and "Finish" not in action:
        i += 1  # Increment round counter
        # print(f"Round {i} - Generating response with the current prompt. Prompt: {prompt}")

        # Request the LLM to generate a Thought and Action
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.5
        )

        # Extract the response content
        response = completion.choices[0].message.content
        print(f"{response.strip()}")  # Print the response per round
        thought, action = parse_response(response)
        # print(f"Thought: {thought}\nAction: {action}")

        # Break if action is Finish
        if "Finish" in action:
            answer_content = action[action.find("[")+1:action.find("]")]
            answer_list = answer_content.split('|')
            # print("Answer List: ", answer_list)
            # print("Observation List: ", observation_list)
            final_answer = generate_final_answer(answer_list=answer_list, observations=observation_list)
            print("Final Response : \n", final_answer)
            break

        # Handle Search or Generate actions
        if "Search" in action:
            # print("Performing search based on the action...")
            entity = action.split("[")[1].rstrip("]")
            print(f"Searching for entity: {entity}")
            search_results = neo4j_conn.find_similar_entities_with_relationships(entity)
            # print(f"Search Results for {entity}: {search_results}")
            top_relationships = select_top_relationships(search_results,thought)
            # print(f"Top Relationships: {top_relationships}")
            observation = top_relationships
            observation_list.append(observation)
            print(f"Observation: {observation}\n")
            
        elif "Generate" in action:
            sub_question = action.split('[')[1].rstrip(']')
            print(f"Generating information for: {sub_question}")

            # Generate a new triple based on the current thought and the context provided by the sub-question
            generated_triple = generate_new_triple(thought + " " + sub_question, client)
            # print(f"Generated Triple: {generated_triple}")

            # Add the generated triple to the Neo4j graph
            result = add_triple_to_graph(neo4j_conn, generated_triple)
            # print(result)

            # Form the observation from generated data
            observation = f"Newly generated triple: {generated_triple}"
            observation_list.append(observation)
            print(f"Generated Observation: {observation}\n")
            
        
        # Update the prompt for the next round of interaction
        prompt += f"\nThought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {observation}"


    return response.strip()

# def select_top_relationships(relationships, thought):
#     """
#     Use the LLM to select the top three (or fewer, depending on availability) most relevant relationships based on the thought.


#     :param relationships: List of all relationships fetched for the top entity.
#     :param previous_thought: The last thought processed, to provide context.
#     :return: A list of top 3 relevant relationships in the form "entity-relationship-entity".
#     """
#     # Construct a prompt for the LLM to evaluate relationships
#     prompt = f"Based on the context: '{thought}', which of the following relationships are most relevant? We need to select 3 triples that will help us the most in trying to answer the question or thought given in the context. Please select the most relevant 3 triples out the ones given below:\n"
#     prompt += "\n".join([f"{i + 1}. {rel['n.name']} - {rel['relationship']} - {rel['m.name']}" for i, rel in enumerate(relationships)])
#     # print(f"Prompt for relationship selection: {prompt}")
#     try:
#         # Using OpenAI's API to get a response based on the prompt
#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=1024,
#             temperature=0.5
#         )

#         # Extracting and interpreting the LLM's response
#         response = completion.choices[0].message.content
#         print(f"Response from LLM: {response}")
#         top_three_indices = parse_llm_response(response, len(relationships))
#         return [f"{relationships[i]['n.name']} - {relationships[i]['relationship']} - {relationships[i]['m.name']}" for i in top_three_indices if i < len(relationships)]
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return []

def select_top_relationships(relationships, thought):
    """
    Use the LLM to select the top three (or fewer, depending on availability) most relevant relationships based on the previous thought.
    """
    # Construct a prompt for the LLM to evaluate relationships
    prompt = f"Based on the context: '{thought}', which of the following relationships are most relevant? We need to select up to 3 triples (or fewer, depending on availability) that will help us the most in trying to answer the question or thought given in the context. Please select the most relevant triples ONLY out the ones given below, DO NOT CREATE YOUR OWN:\n"
    prompt += "\n".join([f"{i + 1}. {rel['n.name']} - {rel['relationship']} - {rel['m.name']}" for i, rel in enumerate(relationships)])
    # print(f"Prompt for relationship selection: {prompt}") 
    # Use only as many selections as there are relationships
    num_relationships = min(len(relationships), 3)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.5
        )

        response = completion.choices[0].message.content
        # print(f"Response from LLM: {response}")
        top_indices = parse_llm_response(response, len(relationships))

        # Ensure that only valid indices are used
        return [f"{relationships[i]['n.name']} - {relationships[i]['relationship']} - {relationships[i]['m.name']}" for i in top_indices if i < len(relationships)]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# def parse_llm_response(response, num_relationships):
#     """
#     Parses the LLM's textual response to extract indices of the top 3 relationships.

#     :param response: Textual response from the LLM containing indices or descriptions.
#     :param num_relationships: Total number of relationships provided to the LLM.
#     :return: A list of indices corresponding to the top three relationships.
#     """
#     import re
#     # Regex to find indices or parse descriptive responses
#     pattern = re.compile(r'\d+')
#     indices = pattern.findall(response)
#     indices = [int(index) - 1 for index in indices if index.isdigit() and int(index) - 1 < num_relationships]
#     return indices[:3]  # Ensure only the top three are selected

def parse_llm_response(response, num_relationships):
    """
    Parses the LLM's textual response to extract indices of the top relationships, up to the number available.

    :param response: Textual response from the LLM containing indices or descriptions.
    :param num_relationships: Total number of relationships provided to the LLM.
    :return: A list of indices corresponding to the top relationships, up to three or fewer if less are available.
    """
    import re
    # Use regular expression to find all numeric indices in the response
    pattern = re.compile(r'\d+')
    indices = pattern.findall(response)

    # Convert extracted indices to integer and adjust for 0-based index, filter out-of-range indices
    valid_indices = [int(index) - 1 for index in indices if int(index) - 1 < num_relationships]

    # Return up to three indices or the number of valid indices, whichever is lesser
    return valid_indices[:min(3, len(valid_indices))]



def format_observations(search_results):
    """ Format search results into an observation string. """
    return "; ".join([f"{rel['n.name']} - {rel['relationship']} - {rel['m.name']}" for rel in search_results])


# def parse_response(response):
#     # Split the response into lines
#     lines = response.strip().split("\n")
#     # Default values if parsing fails
#     thought = "No thought parsed"
#     action = "No action parsed"

#     # Assuming the thought and action are always on specific lines, adjust these indices as necessary
#     if len(lines) > 0:
#         thought = lines[0]
#     if len(lines) > 1:
#         action = lines[1]

#     # Debug output to verify parsing correctness
#     print(f"Debug Parsing - Thought: {thought}, Action: {action}")

#     return thought, action

def parse_response(response):
    """ Parse the response to extract the thought and action. """
    lines = response.split('\n')
    thought = next((line for line in lines if "Thought" in line), None)
    action = next((line for line in lines if "Action" in line), "Finish")

    # Ensure we extract clean thought and action lines
    if thought:
        thought = thought.split("Thought")[1].strip()
    if action:
        action = action.split("Action")[1].strip()

    return thought, action

def perform_search(entity):
    """
    Simulated function to perform a search based on an entity.
    Returns a list of simulated search results.
    """
    return [{"entity": entity, "relation": "related_to", "target": "Result"}]

def format_search_results(results):
    """
    Formats search results into a string for observation.
    """
    return "; ".join(f"{result['entity']} {result['relation']} {result['target']}" for result in results)



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





