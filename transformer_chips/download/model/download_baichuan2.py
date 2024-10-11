# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", trust_remote_code=True)
tokenizer.save_pretrained('./model/Baichuan2-7B')
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", trust_remote_code=True)
model.save_pretrained('./model/Baichuan2-7B')

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Base", trust_remote_code=True)
tokenizer.save_pretrained('./model/Baichuan2-13B')
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Base", trust_remote_code=True)
model.save_pretrained('./model/Baichuan2-13B')