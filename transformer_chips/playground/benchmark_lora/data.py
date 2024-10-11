def process_MMLU(examples):
    prompts, queries, labels = [], [], []
    choices = ["A", "B", "C", "D"]
    def gen_MMLU_prompt(example, include_answer=True):
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

    for i in range(len(examples['question'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_MMLU_prompt(d, include_answer=True)
        prompts.append(prompt)
        queries.append(query)
        labels.append(choices[examples['answer'][i]])

    return {
        'prompt': prompts,
        'query': queries,
        'label': labels,
    }


def process_race(examples):
    prompts, queries, labels = [], [], []
    choices = ["A", "B", "C", "D"]
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

    for i in range(len(examples['answer'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_race_prompt(d, include_answer=True)
        prompts.append(prompt)
        queries.append(query)
        labels.append(examples['answer'][i])

    return {
        'prompt': prompts,
        'query': queries,
        'label': labels,
    }
    

def process_boolq(examples):
    choices = ['no', 'yes']

    prompts, queries, labels = [], [], []
    def gen_boolq_prompt(example, include_answer=True):
        prompt = example["passage"]
        prompt = prompt + '\n\nQuestion: ' + example['question']
        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{choices[int(examples['answer'][i])]}\n\n"
        
        return prompt, query

    for i in range(len(examples['answer'])):
        d = {k: examples[k][i] for k in examples.keys()}
        prompt, query = gen_boolq_prompt(d, include_answer=True)
        prompts.append(prompt)
        queries.append(query)
        labels.append(choices[int(examples['answer'][i])])

    return {
        'prompt': prompts,
        'query': queries,
        'label': labels,
    }
    

def process_c3(examples):
    choices = ["A", "B", "C", "D"]
    def gen_c3_prompt(example, include_answer=True):
        prompt_prefix = f'Article: '

        prompt = prompt_prefix + '\n'.join(example["documents"])
        prompt = prompt + '\n\nQuestion: ' + example['question']
        for i, choice in enumerate(example['choice']):
            prompt += f"\n{choices[i]}. {choice}"

        prompt += "\nAnswer: "
        query = prompt
        if include_answer:
            prompt += f"{example['label']}\n\n"
        
        return prompt, query

    prompts, queries, labels = [], [], []
    for i in range(len(examples['documents'])):
        d = {k: examples[k][i] for k in examples.keys()}
        for j in range(len(d['questions']['answer'])):
            converted_dict = {
                'documents': d['documents'],
                'question': d['questions']['question'][j],
                'choice': d['questions']['choice'][j],
                'label': choices[d['questions']['choice'][j].index(d['questions']['answer'][j])],
            }
            prompt, query = gen_c3_prompt(converted_dict, include_answer=True)
            queries.append(query)
            prompts.append(prompt)
            labels.append(converted_dict['label'])

    return {
        'prompt': prompts,
        'query': queries,
        'label': labels,
    }