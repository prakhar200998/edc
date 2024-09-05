import logging
import os
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
# from query_processor import handle_query  # Assuming this is still needed
from utils import extract_response_new
from entropy import fetch_response, identify_high_entropy_tokens, extract_triple_and_generate_question

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tracking variables
total_time = 0
action_counts = {'Search': 0, 'Generate': 0, 'Finish': 0}
api_call_count = 0
cost_per_api_call = 0.01  # Assuming a cost of $0.01 per API call

# Initialize the sentence transformer model for calculating semantic similarity
# model = SentenceTransformer('all-MiniLM-L6-v2')

def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        global total_time
        total_time += elapsed_time
        return result
    return wrapper

@track_time
def process_question(question, instruction, example, context):
    global api_call_count

    # Combine context with the question for each prompt
    prompt_with_context = context + "\n\n" + question

    # Approach 1: Direct answer with context
    api_call_count += 1
    direct_response_mid = fetch_response(prompt_with_context)
    direct_response = direct_response_mid.choices[0].message.content

    # Approach 2: High entropy token identification and further processing
    api_call_count += 1
    response_mid = fetch_response(prompt_with_context)
    highest_entropy_token = identify_high_entropy_tokens(response_mid)
    triple, generated_question = extract_triple_and_generate_question(response_mid, highest_entropy_token)
    api_call_count += 1
    response_final = extract_response_new(generated_question, instruction, example)
    
    # Track action type
    action_counts['Generate'] += 1

    # Approach 3: Direct to extract_response_new
    api_call_count += 1
    response_direct = extract_response_new(question, instruction, example)

    # Function to process the response (used for both response_final and response_direct)
    def process_response(response):
        if "Thought" in response and "Action" in response:
            # Extract the thought text
            thought_text = response.split("Thought")[1].split("Action")[0].strip()
            
            # Remove any leading numbers and colons from the thought text
            thought_text = thought_text.split(":", 1)[-1].strip()
            
            # Extract the content inside the brackets from the Action part
            action_content = response.split("Action")[1].strip()
            bracket_content = ""
            if "[" in action_content and "]" in action_content:
                bracket_content = action_content.split("[")[1].split("]")[0].strip()

            # Track action type
            if "Generate" in action_content:
                action_counts['Generate'] += 1
            elif "Search" in action_content:
                action_counts['Search'] += 1
            elif "Finish" in action_content:
                action_counts['Finish'] += 1

            # Append the content inside brackets to the processed response
            if bracket_content:
                if not thought_text.endswith('.'):
                    thought_text += '.'
                thought_text += f" {bracket_content}."

            return thought_text.strip()
        else:
            return response.strip()

    # Process responses before returning
    processed_direct_response = process_response(direct_response)
    processed_response_final = process_response(response_final)
    processed_response_direct = process_response(response_direct)

    return processed_direct_response, processed_response_final, processed_response_direct


def main():
    try:
        instruction = """
        Solve a question answering task with interleaving Thought, Action, Observation steps...
        """
        example = """
        Example 1: What types of treatments are available for breast cancer?
        ...
        """
        simplified_context = """
        The questions pertain to the SQuAD dataset with the following titles: 
        ...
        """

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # File paths
        questions_file = os.path.join(script_dir, "../../datasets/SQuAD/selected_squad_questions_new.txt")
        ground_truth_file = os.path.join(script_dir, "../../datasets/SQuAD/selected_squad_answers_new.txt")
        combined_output_file = os.path.join(script_dir, "combined_results.txt")
        answer_file_1 = os.path.join(script_dir, "results_direct_context.txt")
        answer_file_2 = os.path.join(script_dir, "results_high_entropy.txt")
        answer_file_3 = os.path.join(script_dir, "results_direct_extract.txt")

        # Read questions and ground truth answers
        with open(questions_file, "r") as q_file, \
             open(ground_truth_file, "r") as gt_file, \
             open(combined_output_file, "w") as combined_file, \
             open(answer_file_1, "w") as res_file_1, \
             open(answer_file_2, "w") as res_file_2, \
             open(answer_file_3, "w") as res_file_3:

            questions = q_file.readlines()
            ground_truths = gt_file.readlines()
            total_questions = min(2, len(questions))  # Limit to the first 10 questions

            for i in range(total_questions):
                question = questions[i].strip()
                ground_truth = ground_truths[i].strip()
                logging.info(f"Processing Question {i+1}: {question}")
                
                direct_response, response_final, response_direct = process_question(
                    question, instruction, example, simplified_context)

                # Write the processed responses to their respective files
                res_file_1.write(f"{direct_response}\n")
                res_file_2.write(f"{response_final}\n")
                res_file_3.write(f"{response_direct}\n")

                # Write to the combined output file in the desired format
                combined_file.write(f"Q: {question}\n")
                combined_file.write(f"Ground truth: {ground_truth}\n")
                combined_file.write(f"Direct_Context: {direct_response}\n")
                combined_file.write(f"Direct_Extract: {response_direct}\n")
                combined_file.write(f"High_Entropy: {response_final}\n\n")

                logging.info(f"Question {i+1} processed. Combined output and individual files written.")

        # Displaying the collected stats
        total_cost = api_call_count * cost_per_api_call
        print(f"Total Time Taken: {total_time:.2f} seconds")
        print(f"API Calls: {api_call_count}")
        print(f"Total Cost: ${total_cost:.2f}")
        print("Action Counts:", action_counts)

        # Plotting the statistics
        plot_statistics()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

def plot_statistics():
    # Time taken
    plt.figure()
    plt.title("Time Taken for Processing")
    plt.bar(["Total Time"], [total_time])
    plt.ylabel("Time (seconds)")
    plt.savefig("time_taken.png")

    # Action counts
    plt.figure()
    plt.title("Action Distribution")
    plt.bar(action_counts.keys(), action_counts.values())
    plt.ylabel("Number of Actions")
    plt.savefig("action_distribution.png")

    # API call costs
    plt.figure()
    plt.title("Total API Call Costs")
    total_cost = api_call_count * cost_per_api_call
    plt.bar(["Total Cost"], [total_cost])
    plt.ylabel("Cost (USD)")
    plt.savefig("api_call_costs.png")

if __name__ == "__main__":
    main()
