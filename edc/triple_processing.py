import openai
from openai import OpenAI 
import json
import ast

# Set up OpenAI API Key
client = OpenAI(api_key='sk-HX-vp8VFHI2TgNV2JnvxewFiwjL_a6Lf3MeLZN8vCtT3BlbkFJUAoyckTqHcD8n0ECYoRogxarw6QNE0mrPrt9mow3EA')

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
    - Predicates should be uppercase with underscores between words if there are multiple words in the predicate, instead of spaces.

    Based on these guidelines, please provide the correctly formatted triple.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo-16k" if you prefer the turbo models
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
    
    # Calculate the total number of lines in the input file
    with open(input_filename, 'r') as file:
        total_lines = sum(1 for _ in file)
    print(f"Total lines to process: {total_lines}")
    
    line_count = 0
    triple_count = 0
    
    with open(input_filename, 'r') as file:
        for line in file:
            line_count += 1
            # Safe evaluation of the string to a Python literal list
            try:
                triple_list = ast.literal_eval(line.strip())
                for triple in triple_list:
                    triple_count += 1
                    s, p, o = triple
                    fs, fp, fo = format_triple(s.strip(), p.strip(), o.strip())
                    advice = consult_llm(fs, fp, fo)
                    formatted_triples.append(advice)
                    
                    print(f"Processed {triple_count} triples")
            except (SyntaxError, ValueError):
                continue  # Skip lines that cannot be parsed into triples
            
            print(f"Processed {line_count} lines")
    
    # Writing output to a new file
    with open(output_filename, 'w') as outfile:
        for triple in formatted_triples:
            outfile.write(triple + '\n')

# Replace 'path_to_your_file.txt' with your file's path
# Specify the output file path
file_to_process = '../output/squad_schema_canonicalization/edc_output.txt'
processed_file = '../output/squad_schema_canonicalization/edc_output_processed.txt'
process_file(file_to_process, processed_file)
