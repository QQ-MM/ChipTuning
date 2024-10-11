from transformers import AutoProcessor, AutoModelForPreTraining

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-1.5-7b-hf")

processor.save_pretrained('./model/llava-1.5-7b-hf')
model.save_pretrained('./model/llava-1.5-7b-hf')

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-1.5-13b-hf")

processor.save_pretrained('./model/llava-1.5-13b-hf')
model.save_pretrained('./model/llava-1.5-13b-hf')