def remove_empty_lines(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for line in lines:
                if line.strip():  # Check if the line is not empty or whitespace
                    outfile.write(line)
        
        print(f"Empty lines successfully removed and saved to {output_filename}")
    except Exception as e:
        print(f"Error processing the file: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "Chemotherapy.txt"
    output_file = "output_c.txt"
    remove_empty_lines(input_file, output_file)
