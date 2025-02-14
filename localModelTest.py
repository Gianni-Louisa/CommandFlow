import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use a smaller Gemma model (update the identifier as needed)
model_name = "google/gemma-2-2b"  # Replace with your Gemma model's identifier if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Command extraction session started! Type 'exit' to quit.\n")

while True:
    user_input = input("User: ")
    if user_input.strip().lower() == "exit":
        print("Exiting session. Goodbye!")
        break

    prompt = (
        "You are an expert in Windows commands. Translate the following natural language instruction "
        "into the corresponding Windows command.\n\n"
        "Example:\n"
        "Input: 'I want to close a window'\n"
        "Output: 'ALT+F4'\n\n"
        f"Input: '{user_input}'\n"
        "Output:"
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Clear any cached memory before generation
    torch.cuda.empty_cache()
    
    # Wrap in no_grad to save memory
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=80)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the text after the "Output:" marker if present
    if "Output:" in result:
        result = result.split("Output:")[-1].strip()
    
    print("Extracted command:", result)