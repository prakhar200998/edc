from query_processor import handle_query

def main():
    try:
        while True:
            user_query = input("Enter your query or type 'exit' to quit: ")
            if user_query.lower() == 'exit':
                break
            response = handle_query(user_query)
            print("Response:\n", response)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
