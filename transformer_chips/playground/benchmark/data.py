import copy

def gen_MMLU_prompt(example, include_answer=True):
    choices = ["A", "B", "C", "D"]

    subject = example['subject']
    if (subject != ''):
        prompt_prefix = f'The following are multiple choice questions (with answers) about {subject}.\n\n'
    else:
        prompt_prefix = f'The following are multiple choice questions (with answers).\n\n'

    prompt = prompt_prefix + example['question']
    for i, choice in enumerate(example['choices']):
        prompt += f"\n{choices[i]}. {choice}"

    prompt += "\nAnswer:"
    query = prompt
    if include_answer:
        prompt += f" {choices[example['answer']]}\n\n"
    
    return prompt, query


def _tokenize_fn(strings, tokenizer, max_length):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def process_MMLU(examples, tokenizer, include_answer=True):
    prompts, queries = [], []
    for i in range(len(examples['question'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_MMLU_prompt(d, include_answer)
        prompts.append(prompt)
        queries.append(query)

    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length=512) for strings in (prompts, queries)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = -100

    return {
        'input_ids': input_ids,
        'labels': labels,
    }


def process_pg19(examples, tokenizer):
    texts = [text[:20000] for text in examples['text']]
    outputs = tokenizer(
        texts,
        truncation=True,
        return_attention_mask=True,
        max_length=512,
        return_tensors='pt'
    )
    input_ids = outputs.input_ids
    labels = copy.deepcopy(input_ids)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': outputs.attention_mask,
    }


def process_MMLU_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    queries, labels = [], []
    for i in range(len(examples['question'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_MMLU_prompt(d, include_answer=False)
        queries.append(query)
        if (include_answer):
            labels.append(examples['answer'][i])

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }
    
def process_hellaswag_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    def gen_hellaswag_prompt(example, include_answer=True):
        choices = ["A", "B", "C", "D"]

        subject = example['activity_label']
        if (subject != ''):
            prompt_prefix = f'Choose the right ending to complete the following sentence about {subject}.\n\n'
        else:
            prompt_prefix = f'Choose the right ending to complete the following sentence.\n\n'

        ctx = "Question: " + example["ctx_a"] + " " + example["ctx_b"].capitalize()
        prompt = prompt_prefix + ctx
        for i, choice in enumerate(example['endings']):
            prompt += f"\n{choices[i]}. {choice}"

        prompt += "\nAnswer:"
        query = prompt
        if include_answer:
            prompt += f" {choices[example['label']]}\n\n"
        
        return prompt, query

    queries, labels = [], []
    for i in range(len(examples['label'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_hellaswag_prompt(d, include_answer=False)
        queries.append(query)
        if (include_answer):
            labels.append(int(examples['label'][i]))

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }
    

def process_piqa_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    def gen_piqa_prompt(example, include_answer=True):
        choices = ["A", "B"]
        prompt_prefix = f'The following are multiple choice questions (with answers).\n\nQuestion: '

        prompt = prompt_prefix + example["goal"]
        choices = [example["sol1"], example["sol2"]]
        for i, choice in enumerate(choices):
            prompt += f"\n{choices[i]}. {choice}"

        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{choices[example['label']]}\n\n"
        
        return prompt, query

    queries, labels = [], []
    for i in range(len(examples['label'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_piqa_prompt(d, include_answer=False)
        queries.append(query)
        if (include_answer):
            labels.append(int(examples['label'][i]))

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }


def process_race_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    choices = ["A", "B", "C", "D"]
    rev_choices = {"A": 0, "B": 1, "C": 2, "D": 3}
    def gen_race_prompt(example, include_answer=True):
        prompt_prefix = f'Article: '

        prompt = prompt_prefix + example["article"]
        prompt = prompt + '\n\nQuestion: ' + example['question']
        for i, choice in enumerate(example['options']):
            prompt += f"\n{choices[i]}. {choice}"

        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{example['answer']}\n\n"
        
        return prompt, query

    queries, labels = [], []
    for i in range(len(examples['answer'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_race_prompt(d, include_answer=False)
        queries.append(query)
        if (include_answer):
            labels.append(rev_choices[examples['answer'][i]])

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }
    

def process_boolq_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    def gen_boolq_prompt(example, include_answer=True):
        prompt = example["passage"]
        prompt = prompt + '\n\nQuestion: ' + example['question']
        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{example['answer']}\n\n"
        
        return prompt, query

    queries, labels = [], []
    for i in range(len(examples['answer'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_boolq_prompt(d, include_answer=False)
        queries.append(query)
        if (include_answer):
            labels.append(int(examples['answer'][i]))

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }
    

def process_c3_multichoice(examples, tokenizer, include_answer=True, max_length=2048):
    choices = ["A", "B", "C", "D"]
    rev_choices = {"A": 0, "B": 1, "C": 2, "D": 3}
    def gen_c3_prompt(example, include_answer=True):
        prompt_prefix = f'Article: '

        prompt = prompt_prefix + '\n'.join(example["documents"])
        prompt = prompt + '\n\nQuestion: ' + example['question']
        for i, choice in enumerate(example['choice']):
            prompt += f"\n{choices[i]}. {choice}"

        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{choices[example['label']]}\n\n"
        
        return prompt, query

    queries, labels = [], []
    for i in range(len(examples['documents'])):
        d = {k: examples[k][i] for k in examples.keys()}
        for j in range(len(d['questions']['answer'])):
            converted_dict = {
                'documents': d['documents'],
                'question': d['questions']['question'][j],
                'choice': d['questions']['choice'][j],
                'label': d['questions']['choice'][j].index(d['questions']['answer'][j])
            }
            prompt, query = gen_c3_prompt(converted_dict, include_answer=False)
            queries.append(query)
            if (include_answer):
                labels.append(converted_dict['label'])

    examples_tokenized = _tokenize_fn(queries, tokenizer, max_length=max_length)
    input_ids = examples_tokenized["input_ids"]
    probe_pos = [x-1 for x in examples_tokenized['input_ids_lens']]

    if (include_answer):
        return {
            'input_ids': input_ids,
            'labels': labels,
            'probe_pos': probe_pos,
        }
    else:
        return {
            'input_ids': input_ids,
            'probe_pos': probe_pos,
        }