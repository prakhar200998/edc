import openai
from openai import OpenAI 
import json
import ast

# Set up OpenAI API Key
openai.api_key = ''

def format_triple(subject, predicate, object):
    # Apply formatting rules
    formatted_subject = ''.join(word.capitalize() for word in subject.split())
    formatted_predicate = predicate.replace(' ', '_').upper()
    formatted_object = ''.join(word.capitalize() for word in object.split())
    return formatted_subject, formatted_predicate, formatted_object

def consult_llm(subject, predicate, object):
    # Provide detailed context for LLM to optimize formatting
    prompt = f"""
    We need to format triples for a Neo4j graph database, ensuring optimal querying capabilities. Each element of the triples—subject, predicate, and object—must be adapted to meet specific standards suitable for graph querying.

    Current triple:
    - Subject: "{subject}"
    - Predicate: "{predicate}"
    - Object: "{object}"

    Requirements:
    - Normalize spaces, hyphens, numbers, percentages, and decimals.
    - Subject and object should be in CamelCase without special characters except necessary numbers.
    - Predicates should be uppercase with underscores instead of spaces.

    Based on these guidelines, please provide the correctly formatted triple.
    """
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # or "gpt-3.5-turbo-16k" if you prefer the turbo models
            messages=[
                {"role": "system", "content": "You are assisting in formatting data for a Neo4j graph."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        formatted_response = response.choices[0].message.content
        return formatted_response
    except Exception as e:
        print(f"Error during API call: {e}")
        return subject + " " + predicate + " " + object  # Fallback to a simple concatenation

def process_file(input_filename, output_filename):
    formatted_triples = []
    with open(input_filename, 'r') as file:
        for line in file:
            # Safe evaluation of the string to a Python literal list
            try:
                triple_list = ast.literal_eval(line.strip())
                for triple in triple_list:
                    s, p, o = triple
                    fs, fp, fo = format_triple(s.strip(), p.strip(), o.strip())
                    advice = consult_llm(fs, fp, fo)
                    formatted_triples.append(advice)
            except (SyntaxError, ValueError):
                continue  # Skip lines that cannot be parsed into triples

    # Writing output to a new file
    with open(output_filename, 'w') as outfile:
        for triple in formatted_triples:
            outfile.write(triple + '\n')

# Replace 'path_to_your_file.txt' with your file's path
# Specify the output file path
process_file('edc_output.txt', 'edc_output_processed.txt')
