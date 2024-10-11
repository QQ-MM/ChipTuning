def process(queries, images, processor, max_length):
    """Tokenize a list of multimodal inputs."""
    tokenized_list = [
        processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text, image in zip(queries, images)
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    pixel_values = [tokenized.pixel_values[0] for tokenized in tokenized_list]
    attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(processor.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        input_ids_lens=input_ids_lens,
        attention_mask=attention_mask,
    )
    

def process_flowers_multichoice(examples, processor, include_answer=True, max_length=2048, image_seq_length=576):
    def gen_flowers_prompt():
        conversation_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What type of object is in this photo?"},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation_template, add_generation_prompt=True)
        return text_prompt

    queries, images, labels = [], [], []
    for i in range(len(examples['label'])):
        query = gen_flowers_prompt()
        queries.append(query)
        images.append(examples['image'][i])
        if (include_answer):
            labels.append(examples['label'][i])

    examples_tokenized = process(queries, images, processor, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    pixel_values = examples_tokenized["pixel_values"]
    attention_mask = examples_tokenized["attention_mask"]
    probe_pos = [(x-1)+image_seq_length-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }

    
def process_caltech_multichoice(examples, processor, include_answer=True, max_length=2048, image_seq_length=576):
    def gen_caltech_prompt():
        conversation_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What type of object is in this photo?"},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation_template, add_generation_prompt=True)
        return text_prompt

    queries, images, labels = [], [], []
    for i in range(len(examples['label'])):
        query = gen_caltech_prompt()
        queries.append(query)
        images.append(examples['image'][i])
        if (include_answer):
            labels.append(examples['label'][i])

    examples_tokenized = process(queries, images, processor, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    pixel_values = examples_tokenized["pixel_values"]
    attention_mask = examples_tokenized["attention_mask"]
    probe_pos = [(x-1)+image_seq_length-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }


def process_stanfordcars_multichoice(examples, processor, include_answer=True, max_length=2048, image_seq_length=576):
    def gen_stanfordcars_prompt():
        conversation_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What type of object is in this photo?"},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation_template, add_generation_prompt=True)
        return text_prompt

    queries, images, labels = [], [], []
    for i in range(len(examples['label'])):
        query = gen_stanfordcars_prompt()
        queries.append(query)
        images.append(examples['image'][i])
        if (include_answer):
            labels.append(examples['label'][i])

    examples_tokenized = process(queries, images, processor, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    pixel_values = examples_tokenized["pixel_values"]
    attention_mask = examples_tokenized["attention_mask"]
    probe_pos = [(x-1)+image_seq_length-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask,
            'probe_pos': probe_pos,
        }