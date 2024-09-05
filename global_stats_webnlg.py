# Accuracy for each system (WebNLG dataset)
accuracy_webnlg = {
    'baseline_LLM': 0.52,
    'EDC_GoG_pipeline': 0.66,
    'integrated_EDC_GoG_refinement': 0.72,
}

# Query Answering (QA) Accuracy for Incomplete graph (100 examples)
qa_accuracy_webnlg = {
    'EDC_pipeline': 0.15,
    'GoG_pipeline': 0.56,
    'integrated_EDC_GoG_refinement': 0.54
}

# Execution Time (in minutes) QA - 500 examples (WebNLG dataset)
execution_time_webnlg = {
    'EDC_Graph_Creation': 140,    
    'QA_GoG': 118,                
    'QA_integrated': 190,          
    'Accuracy_Calculation': 18,
}

# Scalability (Processing Time in minutes for varying data sizes) (10% QA for WebNLG)
scalability_webnlg = {
    'data_size_200': {    
        'EDC_Graph_Creation': 31,
        'QA_GoG': 25,
        'QA_integrated': 40,
        'Accuracy_Calculation': 5,
    },
    'data_size_500': {    
        'EDC_Graph_Creation': 69,
        'QA_GoG': 62,
        'QA_integrated': 94,
        'Accuracy_Calculation': 12,
    },
    'data_size_1000': {   
        'EDC_Graph_Creation': 140,
        'QA_GoG': 118,
        'QA_integrated': 190,
        'Accuracy_Calculation': 18,
    }
}

# Dataset Size for WebNLG (Schema Canonicalization) (1000 lines of text)
dataset_size_webnlg = {
    'nodes': {
        'Schema_Canonicalization': 8423,   
        'Target_Alignment': 8102,  
    },
    'relationships': {
        'Schema_Canonicalization': 3946,   
        'Target_Alignment': 2803,  
    }
}
