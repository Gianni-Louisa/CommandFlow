import openai
import os

# Set your OpenAI API key; you can also set it as an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY", "KEYHERE")

def api_call(messages):
    try:
        print("Sending...")
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0,
            max_tokens=1000,
            frequency_penalty=0,
            presence_penalty=0,
            top_p=1
        )
        print("Done.")
        return response['choices'][0]['message']['content']
    except Exception as error:
        print("Error in api_call:", error)
        raise

def main():
    # Initialize conversation history with the system prompt.
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant that is able take a sentence and return a command that can be used in a windows terminal. You will only return the command, and nothing else. "
        }
    ]

    print("Chat session started! Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Append the user's message to the conversation history.
        conversation.append({"role": "user", "content": user_input})

        # Make the API call with the complete conversation.
        assistant_reply = api_call(conversation)
        print("Assistant:", assistant_reply, "\n")

        # Append the assistant's reply to the conversation history.
        conversation.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
