# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
tokenizer.save_pretrained('./model/Llama-2-7b-hf')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
model.save_pretrained('./model/Llama-2-7b-hf')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", trust_remote_code=True)
tokenizer.save_pretrained('./model/Llama-2-13b-hf')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", trust_remote_code=True)
model.save_pretrained('./model/Llama-2-13b-hf')