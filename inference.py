# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import save_expert_log
import torch

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

prompt = "Who are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(inputs)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

save_expert_log("my_expert_activations.json")
print("Expert activation log saved to my_expert_activations.json")