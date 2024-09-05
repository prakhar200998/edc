# import numpy as np
# from openai import OpenAI
# from config import OPENAI_API_KEY
# import nltk
# from nltk.corpus import stopwords

# # Ensure you have the necessary NLTK data downloaded
# nltk.download('stopwords')

# client = OpenAI(api_key=OPENAI_API_KEY)

# def fetch_response(prompt):
#     """
#     Fetch the generated response using OpenAI's GPT-3.5-turbo model.
#     """
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=1024,
#         logprobs=True,  # Requesting log probabilities for the top 10 tokens
#         top_logprobs=2
#     )
#     return completion

# def print_response_details(response):
#     """
#     Print details of the response including messages and log probabilities.
#     """
#     print("Complete Response JSON:")
#     print(response)
#     print("\nExtracted Message Content:")
#     print(response.choices[0].message.content)  # Using .content to access message text

#     print("\nToken Log Probabilities Details:")
#     if hasattr(response.choices[0], 'logprobs'):
#         logprobs_content = response.choices[0].logprobs.content  # Accessing .content of logprobs directly
#         for token_info in logprobs_content:
#             token = token_info.token  # Direct attribute access
#             logprob = token_info.logprob
#             top_logprobs = [f"{top.token}: {top.logprob}" for top in token_info.top_logprobs]
#             print(f"Token: {token}, LogProb: {logprob}, Top LogProbs: {top_logprobs}")

# def calculate_entropy(token_info):
#     """
#     Calculate the entropy of a token given its top log probabilities.
#     """
#     if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
#         log_probs = [top.logprob for top in token_info.top_logprobs]
#         probs = np.exp(log_probs)
#         entropy = -np.sum(probs * np.log(probs))
#         return entropy
#     return 0

# def identify_high_entropy_tokens(response):
#     """
#     Identify tokens with high entropy in the generated text.
#     """
#     stop_words = set(stopwords.words('english'))  # Load stopwords from NLTK
#     custom_filter = set(["is", "are", "was", "were", "can", "will", "he", "she", "it", "they", "and", "but", "or", "in", "at", "by", "with", "just", "only", "also", "um", "ah", "well"])

#     def is_substantive(token):
#         return token.lower() not in stop_words and token.lower() not in custom_filter

#     if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
#         token_logprobs = response.choices[0].logprobs.content
#         entropies = [calculate_entropy(token) for token in token_logprobs]
#         substantive_tokens = [(token.token, entropy) for token, entropy in zip(token_logprobs, entropies) if is_substantive(token.token) and entropy > np.median(entropies)]

#         # Sort by entropy and optionally print the top N
#         high_entropy_tokens = sorted(substantive_tokens, key=lambda x: x[1], reverse=True)[:3]  # Change 5 to your desired number of top tokens
        
#         print("\nTop High Entropy Substantive Tokens:")
#         for token, entropy in high_entropy_tokens:
#             print(f"Token: {token}, Entropy: {entropy:.2f}")

# # Example usage
# prompt = "What are metastatic colorectal cancer patients treated with? Answer in 1-2 sentences at max."
# response = fetch_response(prompt)
# print_response_details(response)
# identify_high_entropy_tokens(response)

import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY
import nltk
from nltk.corpus import stopwords
import string

# Ensure you have the necessary NLTK data downloaded
nltk.download('stopwords')

client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_response(prompt):
    """
    Fetch the generated response using OpenAI's GPT-4 model.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use a valid model name
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        logprobs=True,  # Requesting log probabilities for the top tokens
        top_logprobs=2
    )
    return completion

def calculate_entropy(token_info):
    """
    Calculate the entropy of a token given its top log probabilities.
    """
    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
        log_probs = [top.logprob for top in token_info.top_logprobs]
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    return 0

def identify_high_entropy_tokens(response):
    stop_words = set(stopwords.words('english'))
    custom_filter = set(["is", "are", "was", "were", "can", "will", "he", "she", "it", "they", "and", "but", "or", "in", "at", "by", "with", "just", "only", "also", "um", "ah", "well", "if", "used"])
    additional_words = set(["may", "might", "could", "would", "should", "shall", "which", "who", "whom", "whose", "so", "yet", "for", "nor", "a", "an", "the", "this", "that", "these", "those", "have", "has", "had"])

    # Combine all filters into one set for easier management
    all_filters = stop_words.union(custom_filter, additional_words)

    if hasattr(response.choices[0], 'logprobs'):
        token_logprobs = response.choices[0].logprobs.content
        entropies = [(token.token, calculate_entropy(token)) for token in token_logprobs]

        # Filtering substantive tokens
        substantive_tokens = [(token, entropy) for token, entropy in entropies if token.lower() not in all_filters and token not in string.punctuation and not token.isdigit()]

        # Sort by entropy
        high_entropy_tokens = sorted(substantive_tokens, key=lambda x: x[1], reverse=True)[:3]

        print("\nTop High Entropy Substantive Tokens:")
        for token, entropy in high_entropy_tokens:
            print(f"Token: {token}, Entropy: {entropy:.2f}")

        if high_entropy_tokens:
            return high_entropy_tokens[0][0]  # Return the token with the highest entropy
    return None

def extract_triple_and_generate_question(response, high_entropy_token):
    """
    Extract a meaningful triple and generate a verification question using the LLM,
    and return them as separate variables.
    """
    message_content = response.choices[0].message.content.strip()
    
    # Define the prompt for the LLM
    prompt = (
        f"Given the following text, extract a triple (subject, predicate, object) where '{high_entropy_token}' "
        f"is part of the information and formulate a question to verify the extracted triple. The question "
        f"should be open-ended and not focused on the specific object:\n\n"
        f"Text: {message_content}\n\n"
        f"Triple format: (subject, predicate, object)\n\n"
        f"Example 1: Text: 'The Eiffel Tower is in Paris.'\nTriple: ('Eiffel Tower', 'is in', 'Paris')\n"
        f"Question: 'Where is the Eiffel Tower located?'\n\n"
        f"Example 2: Text: 'Water boils at 100 degrees Celsius.'\nTriple: ('Water', 'boils at', '100 degrees Celsius')\n"
        f"Question: 'At what temperature does water boil?'\n\n"
        f"Triple and Question:"
    )
    
    # Call the OpenAI API to execute the prompt
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this is the correct model name you have access to
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    # Retrieve and process the result to separate the triple and the question
    output_text = completion.choices[0].message.content.strip()
    triple_start = output_text.find("Triple: ") + len("Triple: ")
    question_start = output_text.find("Question: ") + len("Question: ")
    triple_end = output_text.find("\nQuestion: ")
    
    if triple_start > -1 and question_start > -1 and triple_end > -1:
        triple_text = output_text[triple_start:triple_end].strip()
        question_text = output_text[question_start:].strip()
    else:
        triple_text = "No triple found."
        question_text = "No question generated."

    return triple_text, question_text

def main():
    # user_input = input("Enter your prompt: ")
    # prompt = user_input+"Answer in 1-2 sentences at max."
    # response = fetch_response(prompt)
    
    # print("Response Details:")
    # print(response.choices[0].message.content.strip())
    
    # # Identify the token with the highest entropy
    # highest_entropy_token = identify_high_entropy_tokens(response)
    # print(f"Highest Entropy Token: {highest_entropy_token}")
    
    # # Extract a triple and generate a question using LLM
    # triple, question = extract_triple_and_generate_question(response, highest_entropy_token)
    # print(f"Triple: {triple}")
    # print(f"Question: {question}")
    pass

if __name__ == "__main__":
    main()
