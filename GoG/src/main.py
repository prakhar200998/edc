from query_processor import handle_query
from utils import extract_response_new

def main():
    try:

        instruction = """
        Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the 3 most similar entities on the graph and returns their one-hop subgraphs.
        (2) Generate[thought], which generate some new triples related to your last thought.
        (3) Finish[answer1 | answer2 | ...], which returns the answer and finishes the task.
        """

        example = """
        Example 1: What types of treatments are available for breast cancer?
        Topic Entity: [Breast Cancer]
        Thought 1: I need to search for 'Breast Cancer' to find the treatments associated with it.
        Action 1: Search[Breast Cancer]
        Observation 1: Breast Cancer, treated_with, Chemotherapy; Breast Cancer, treated_with, Radiation Therapy
        Thought 2: Both chemotherapy and radiation therapy are noted. I should check what specific drugs are used in chemotherapy for breast cancer.
        Action 2: Search[Chemotherapy for Breast Cancer]
        Observation 2: Chemotherapy for Breast Cancer, uses_drug, Taxol; Chemotherapy for Breast Cancer, uses_drug, Adriamycin
        Thought 3: Taxol and Adriamycin are commonly used. I should provide this information.
        Action 3: Finish[Taxol | Adriamycin]
        """

        question = "What are the side effects of Chemotherapy?"

        
        
        
        # Call the function with these parameters
        response = extract_response_new(question, instruction, example)
        print(response)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
