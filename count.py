import pyarrow as pa

# Path to the Arrow file
file_path = 'tekgen/output/dataset/train/data-00000-of-00001.arrow'

# Attempt to load the Arrow file
try:
    with pa.memory_map(file_path, 'r') as source:
        # Try reading as an IPC file
        table = pa.ipc.open_file(source).read_all()
except pa.lib.ArrowInvalid:
    # If not an IPC file, try reading as an IPC stream
    with pa.memory_map(file_path, 'r') as source:
        table = pa.ipc.open_stream(source).read_all()

# Print the number of records
print("Number of records:", table.num_rows)
