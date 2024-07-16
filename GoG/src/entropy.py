import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY
import nltk
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data downloaded
nltk.download('stopwords')

client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_response(prompt):
    """
    Fetch the generated response using OpenAI's GPT-3.5-turbo model.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        logprobs=True,  # Requesting log probabilities for the top 10 tokens
        top_logprobs=2
    )
    return completion

def print_response_details(response):
    """
    Print details of the response including messages and log probabilities.
    """
    print("Complete Response JSON:")
    print(response)
    print("\nExtracted Message Content:")
    print(response.choices[0].message.content)  # Using .content to access message text

    print("\nToken Log Probabilities Details:")
    if hasattr(response.choices[0], 'logprobs'):
        logprobs_content = response.choices[0].logprobs.content  # Accessing .content of logprobs directly
        for token_info in logprobs_content:
            token = token_info.token  # Direct attribute access
            logprob = token_info.logprob
            top_logprobs = [f"{top.token}: {top.logprob}" for top in token_info.top_logprobs]
            print(f"Token: {token}, LogProb: {logprob}, Top LogProbs: {top_logprobs}")

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
    """
    Identify tokens with high entropy in the generated text.
    """
    stop_words = set(stopwords.words('english'))  # Load stopwords from NLTK
    custom_filter = set(["is", "are", "was", "were", "can", "will", "he", "she", "it", "they", "and", "but", "or", "in", "at", "by", "with", "just", "only", "also", "um", "ah", "well"])

    def is_substantive(token):
        return token.lower() not in stop_words and token.lower() not in custom_filter

    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
        token_logprobs = response.choices[0].logprobs.content
        entropies = [calculate_entropy(token) for token in token_logprobs]
        substantive_tokens = [(token.token, entropy) for token, entropy in zip(token_logprobs, entropies) if is_substantive(token.token) and entropy > np.median(entropies)]

        # Sort by entropy and optionally print the top N
        high_entropy_tokens = sorted(substantive_tokens, key=lambda x: x[1], reverse=True)[:3]  # Change 5 to your desired number of top tokens
        
        print("\nTop High Entropy Substantive Tokens:")
        for token, entropy in high_entropy_tokens:
            print(f"Token: {token}, Entropy: {entropy:.2f}")

# Example usage
prompt = "Discuss the impact of quantum computing on cryptography. Answer in a single short sentence only."
response = fetch_response(prompt)
print_response_details(response)
identify_high_entropy_tokens(response)
