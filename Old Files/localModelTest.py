# Filename: localModelTest.py
# Description: Local LLM testing script using Gemma 2 2B model for Windows command extraction.
#              This script loads a local Gemma model and provides a command-line interface
#              for users to input natural language requests, which are then translated into
#              corresponding Windows commands. ChatGPT Assisted.
#
# Programmer: Gianni Louisa
# Date Created: 2/25/2025
# Last Revised: 2/25/2025
#
# Revision History:
#   - 2/25/2025 by Gianni Louisa: Initial creation with basic Gemma model integration.
#
# Preconditions:
#   - Python 3.x installed.
#   - Required packages installed: torch, transformers.
#   - Sufficient system memory and GPU resources (if available) to run the Gemma model.
#
# Acceptable Input Values/Types:
#   - Natural language text describing desired Windows operations.
#
# Unacceptable Input Values/Types:
#   - Empty inputs or inputs unrelated to Windows commands may produce unexpected results.
#
# Postconditions:
#   - The script outputs the extracted Windows command corresponding to the user's request.
#
# Return Values:
#   - This script does not return any values; it prints the extracted commands to the console.
#
# Error/Exception Conditions:
#   - CUDA out of memory errors may occur if the model is too large for available GPU memory.
#   - Network errors may occur during model download if not previously cached.
#
# Side Effects:
#   - Utilizes significant system resources, particularly GPU memory if available.
#
# Invariants:
#   - The model remains loaded throughout the session until the user exits.
#
# Known Faults:
#   - The model may occasionally generate incorrect or incomplete commands.
#   - Large inputs may cause memory issues on systems with limited resources.

# Import required libraries
import torch  # Import PyTorch for tensor operations and GPU acceleration
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import Hugging Face transformers for model loading

# Define the model to be used - Gemma 2 2B is a smaller model suitable for local execution
model_name = "google/gemma-2-2b"  # Specify the model identifier from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer for the specified model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # Load the model with half-precision to reduce memory usage

# Determine the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for CUDA GPU availability
model.to(device)  # Move the model to the appropriate device (GPU or CPU)

# Print welcome message to indicate the session has started
print("Command extraction session started! Type 'exit' to quit.\n")  # Inform user about how to exit the program

# Main interaction loop
while True:  # Continue running until explicitly broken
    user_input = input("User: ")  # Prompt for and capture user input
    if user_input.strip().lower() == "exit":  # Check if user wants to exit
        print("Exiting session. Goodbye!")  # Display exit message
        break  # Exit the loop and end the program

    # Construct the prompt with instructions and examples for the model
    prompt = (
        "You are an expert in Windows commands. Translate the following natural language instruction "  # Define the model's role
        "into the corresponding Windows command.\n\n"  # Specify the task
        "Example:\n"  # Provide an example for few-shot learning
        "Input: 'I want to close a window'\n"  # Example input
        "Output: 'ALT+F4'\n\n"  # Example expected output
        f"Input: '{user_input}'\n"  # Include the actual user input
        "Output:"  # Prompt marker for the model's response
    )
    
    # Convert the text prompt to token IDs and move to the appropriate device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # Tokenize and prepare input for the model
    
    # Free up GPU memory before generation to prevent out-of-memory errors
    torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
    
    # Generate the response without tracking gradients to save memory
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(input_ids, max_new_tokens=80)  # Generate response with a maximum of 80 new tokens
    
    # Decode the generated token IDs back to text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Convert model output tokens to readable text
    
    # Process the result to extract only the relevant command
    if "Output:" in result:  # Check if the output marker exists in the result
        result = result.split("Output:")[-1].strip()  # Extract only the text after the "Output:" marker
    
    # Display the extracted command to the user
    print("Extracted command:", result)  # Print the final extracted command