# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.models.gpt_oss.modeling_gpt_oss import save_expert_log
from transformers.models.gpt_oss.modeling_gpt_oss import EXPERT_LOG
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

input_file = "mkqa.jsonl"
print(f"Starting to process {input_file}...")

with open(input_file, 'r', encoding='utf-8') as f:
    # Process each line (each line is a separate JSON object)
    for i, line in enumerate(f):
        # Parse the JSON data for the current question
        data = json.loads(line)
        queries = data.get("queries", {})
        answers = data.get("answers", {})
        
        # This dictionary will store the expert logs for all language versions of the current question
        question_expert_logs = {}
        
        print(f"\nProcessing Question #{i}...")

        # --- 3. Iterate Through Each Language Version ---
        for lang, prompt in queries.items():
            if lang == "nl":
                break
            # Clear the global EXPERT_LOG before each model run to ensure clean data
            EXPERT_LOG.clear()
            
            print(f"  - Generating for language: '{lang}'")

            # Prepare the inputs for the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Run the model
            # We don't need the generated output text, just the expert logs
            outputs = model.generate(**inputs, max_new_tokens=40)

            prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

            # After generation, EXPERT_LOG is populated.
            # We store a copy of it in our dictionary for this question.
            if EXPERT_LOG:
                # The log contains data for layer 8 as configured in the model forward pass
                question_expert_logs[lang] = {
                    "representations": EXPERT_LOG[:], # Use a copy
                    "pred": prediction,
                    "gt": answers.get(lang, "N/A")
                }
                                              
            else:
                print(f"    - Warning: No expert log was generated for language '{lang}'.")


        # --- 4. Save the Collected Logs to a File ---
        if question_expert_logs:
            output_filename = f"question_{i}_experts.json"
            with open(output_filename, "w", encoding='utf-8') as out_f:
                # The file will contain the logs for all languages of this one question
                json.dump(question_expert_logs, out_f, indent=4)
            print(f"Successfully saved expert logs for Question #{i} to '{output_filename}'")
        else:
            print(f"No expert data was collected for Question #{i}, skipping file save.")

        break

print("\nProcessing complete.")

