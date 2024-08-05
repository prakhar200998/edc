import pyarrow as pa
import pandas as pd

def read_and_print_arrow_stream(filename, num_rows=3):
    with open(filename, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        for batch in reader:
            df = batch.to_pandas()
            pd.set_option('display.max_columns', None)  # Show all columns
            print(df.head(num_rows))  # Print the first few rows
            break  # Remove this line if you want to print more batches

filename = 'dataset/train/data-00000-of-00001.arrow'
read_and_print_arrow_stream(filename)
