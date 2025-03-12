# Filename: apiModelTest.py
# Description: OpenAI API integration script for Windows command extraction.
#              This script connects to OpenAI's API to convert natural language requests
#              into corresponding Windows terminal commands. It maintains a conversation
#              history to provide context for the AI model. ChatGPT Assisted.
#
# Programmer: Gianni Louisa
# Date Created: 2/25/2025
# Last Revised: 2/25/2025
#
# Revision History:
#   - 2/25/2025 by Gianni Louisa: Initial creation with OpenAI API integration.
#
# Preconditions:
#   - Python 3.x installed.
#   - Required packages installed: openai.
#   - Valid OpenAI API key set as environment variable or directly in the script.
#   - Internet connection to access the OpenAI API.
#
# Acceptable Input Values/Types:
#   - Natural language text describing desired Windows operations.
#
# Unacceptable Input Values/Types:
#   - Empty inputs may produce unexpected results.
#
# Postconditions:
#   - The script outputs the extracted Windows command corresponding to the user's request.
#
# Return Values:
#   - This script does not return any values; it prints the extracted commands to the console.
#
# Error/Exception Conditions:
#   - API rate limit errors may occur if too many requests are made in a short period.
#   - Network errors may occur if the internet connection is unstable.
#   - Authentication errors may occur if the API key is invalid or expired.
#
# Side Effects:
#   - Makes external API calls to OpenAI's servers.
#   - Incurs usage costs based on OpenAI's pricing model.
#
# Invariants:
#   - The conversation history maintains the context throughout the session.
#
# Known Faults:
#   - The model may occasionally generate incorrect or incomplete commands.

# Import required libraries
import openai  # Import the OpenAI Python client library for API access
import os  # Import os module for environment variable access

# Set your OpenAI API key; you can also set it as an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY", "KEYHERE")  # Get API key from environment or use default

def api_call(messages):
    """
    Make an API call to OpenAI's ChatCompletion endpoint.
    
    Args:
        messages (list): List of message dictionaries containing role and content.
        
    Returns:
        str: The content of the assistant's response.
        
    Raises:
        Exception: Any error that occurs during the API call.
    """
    try:
        print("Sending...")  # Indicate that the API request is being sent
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",  # Specify the model version to use
            messages=messages,  # Pass the conversation history
            temperature=0,  # Set temperature to 0 for more deterministic outputs
            max_tokens=1000,  # Limit the response length
            frequency_penalty=0,  # No penalty for frequency of token usage
            presence_penalty=0,  # No penalty for presence of tokens
            top_p=1  # Use all tokens in the distribution (no filtering)
        )
        print("Done.")  # Indicate that the API request has completed
        return response['choices'][0]['message']['content']  # Extract and return the response content
    except Exception as error:
        print("Error in api_call:", error)  # Print error details
        raise  # Re-raise the exception for handling by the caller

def main():
    """
    Main function to run the chat session with the OpenAI API.
    """
    # Initialize conversation history with the system prompt.
    conversation = [
        {
            "role": "system",  # System message defines the assistant's behavior
            "content": "You are a helpful assistant that is able take a sentence and return a command that can be used in a windows terminal. You will only return the command, and nothing else. "
        }
    ]

    print("Chat session started! Type 'exit' to quit.\n")  # Display welcome message

    while True:  # Continue running until explicitly broken
        user_input = input("User: ")  # Prompt for and capture user input
        if user_input.strip().lower() == "exit":  # Check if user wants to exit
            print("Exiting chat. Goodbye!")  # Display exit message
            break  # Exit the loop and end the program

        # Append the user's message to the conversation history.
        conversation.append({"role": "user", "content": user_input})  # Add user message to conversation

        # Make the API call with the complete conversation.
        assistant_reply = api_call(conversation)  # Get response from OpenAI API
        print("Assistant:", assistant_reply, "\n")  # Display the assistant's response

        # Append the assistant's reply to the conversation history.
        conversation.append({"role": "assistant", "content": assistant_reply})  # Add assistant response to conversation

if __name__ == "__main__":
    main()  # Execute the main function when the script is run directly
