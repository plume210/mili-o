import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
import json
import os
import numpy as np 
from .data_augmentation import augement_dialogue

class Gen_dataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        output_base: str,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train
        setting: str = None,
        inference: bool=False,
        mem_drop_rate: float = 0,
        history_aware: bool = False
    ):
        self.history_aware = history_aware
        self.mem_drop_rate = mem_drop_rate
        
        self.tokenizer = tokenizer
        self.datasets = {}
        self.quality_levels = {
            'claude': 3,
            # 'openai': 3,
            # 'seventyFull': 2,
            # 'seventyQuantized': 2,
            # 'eightB': 1
        }
        self.inference = inference
        self.sample_size = 0
        self.setting = setting

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = [f'{output_base}/{split_set}/{input_dir}' for input_dir in input_dirs]

        if split_set == 'topic':
            dataset_lists = [f'{output_base}/topic_{topic_num}/claude' for topic_num in self.setting]

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder.split('/')[-2]
            print(f"Loading ... dataset: {dataset_folder}")
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            if dataset_name in self.quality_levels.keys():
                elements['quality'] = self.quality_levels[dataset_name]
            else:
                elements['quality'] = 1
            # if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
            #     continue # skip it
            # with open(f"./mydatasets/{dataset_name}.json", 'r') as file:
            #     data = json.load(file)
            # samples = data[split_set] 
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            # if len(samples) > 4000:
            #     elements['samples'] = samples[:4000]
            # else:
            #     elements['samples'] = samples
            elements['samples'] = samples
            elements['len'] = len(elements['samples'])
            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1].split('/')[-2]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive
            self.datasets[dataset_name] = elements

    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def tokenize_dialogue_label(self, dialogue):
        text = " ".join(dialogue.split('\n'))
        tokens = self.tokenizer.encode(text.strip(), return_tensors='pt')[0]
        return tokens
        
    # Dataset method - sample_size stores the sum of the lengths of all dataset sources
    def __len__(self):
        return self.sample_size

    # calculates the group an index falls into, and returns that group and the local index
    # returns tuple as dataset_name, index
    def shifted_index(self, i):
        # [start_index, end_index) - end is not inclusive
        for group in self.datasets.keys():
            if i < self.datasets[group]['end_index'] and i >= self.datasets[group]['start_index']:
                shifted_index = i - self.datasets[group]['start_index']
                return group, shifted_index
        return -1, -1
            
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        selected_dataset, shifted_i = self.shifted_index(i)
        sample = self.datasets[selected_dataset]['samples'][shifted_i]
        ### reading the dialogue, label and whisper from disk
        dialogue_text = ""
        whisper_text = ""
        # label_text = ""
        # answer_text = ""
        dialogue_text = (Path(sample) / 'dialogue.txt').read_text()
        # label_text = (Path(sample) / 'values.txt').read_text()
        whisper_text = (Path(sample) / 'whisper.txt').read_text()
        memory_text = (Path(sample) / 'memory.txt').read_text()
        # mask_text = (Path(sample) / 'mask.txt').read_text()

        rules = """
        The user has three use cases for a proactive agent:
        1. Reminding.
        2. Social Guidance: Scenarios that warrant social guidance may involve an interview, first date, or public speaking, etc.
        3. Managing emotional dysregulation.

        Here are some guidelines for answering questions:
        1. Your answer must be at most 3 English words long. After your answer is outputted, stop generating.
        2. Your answer must be in accordance with the nine principles for proactive agents.
        3. Your answer must not contain any emotion tokens, or any of "|"
        4. Do not generate anything except for your answer. This includes any notes in parentheses, or explanations.
        5. You are an audio assistant, and you are speaking directly to the user through audio. Only output the answer, nothing else.
        """
        
        starting_rules = """
            You are a proactive AI agent designed to actively help humans by reminding and assisting them in different scenarios, by whispering short, concise phrases (1-3 words) to its user. We define nine principles to guide desired proactive agent behavior.

            - Valuable: advances the user’s interests and tasks, in the user’s opinion
            - Pertinent: attentive to the current situation
            - Competent: within the scope of the agent’s abilities and knowledge
            - Unobtrusive: not interfering with the user’s own activities or attention, without warrant - Transparent: understandable to the user
            - Controllable: exposed to the scrutiny and according to the mandate of the user
            - Deferent: gracefully unimposing
            - Anticipatory: aware of current and future needs and opportunities
            - Safe: minimizes negative consequences, in the user’s opinion
        """

        # bigger_rules = f'{starting_rules}\n{rules}'

        # NOTE settings
        # if self.setting == "IO":
        #     dialogue_text = f"Input:\n---\n{dialogue_text}\n---\nOutput:\n---\n"
        # elif self.setting == "prompt":
        #     dialogue_text = f"You are an assistive agent with a set of rules to follow. You will be presented with a set of rules, a dialogue and the last thing a user said. Infer the user's last question and answer it, within accordance of the rules.\n\n---\nRules:\n{rules}\n\n---dialogue:\n{dialogue_text}\n\n---\nAnswer:\n"
        # elif self.setting == "bigger_prompt":
        #     dialogue_text = f"You are an assistive agent with a set of rules to follow. You will be presented with a set of rules and a dialogue. Infer the user's last question and answer it, within accordance of the rules.\n\n---\nRules:\n{bigger_rules}\n\n---\nDialogue:\n{dialogue_text}\n\n---\nAnswer:\n"
        # elif self.setting == "bigger_no_infer":
        #     dialogue_text = f"You are an assistive agent with a set of rules to follow. You will be presented with a set of rules and a dialogue. Respond to the user within accordance of the rules.\n\n---\nRules:\n{bigger_rules}\n\n---\nDialogue:\n{dialogue_text}\n\n---\nAnswer:\n"

        ## prepare conversation template
        messages = [
            {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
            {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{memory_text}'},
            {"role": "user", "content": dialogue_text},
        ]
        conv_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_text = conv_text + whisper_text + '<|end_of_text|>'
        ## The reason we use '<|end_of_text|>' 
        ## because the default data collator will replace all eos token to ignore_id
        ## so we first use '<|end_of_text|>' to avoid replacement and then in the data collator we replace it with eos id

        # tokenize the dialogue
        input_ids = self.tokenizer.encode(conv_text, return_tensors='pt')[0] #self.tokenize_dialogue_label(conv_text)
        label_ids = self.tokenizer.encode(all_text, return_tensors='pt')[0]
        

        padded_input_ids = torch.clone(label_ids) 

        if self.inference:
            # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)
            output = {
                "input_ids": padded_input_ids,
                "raw_input": input_ids,
                "raw_text": all_text
                # "labels": padded_label_ids,
            }
        else:   
            # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)
            ## the labels will be generated in data collator
            output = {
                "input_ids": padded_input_ids,
                # "raw_input": input_ids,
                # "raw_text": all_text
                # "labels": padded_label_ids,
            }
            
        return output



class Gen_dataset_New(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        dataset_names: list = [],
        dataset_probs: list = [],
        split_set: str = "Train",
        inference: bool=False,
        mem_drop_rate: float = 0,
        neg_prob: float = 0,
        history_aware: bool = False,
        aug_config = None
    ):
        self.history_aware = history_aware
        self.mem_drop_rate = mem_drop_rate
        
        self.tokenizer = tokenizer
        self.datasets = {}

        self.inference = inference
        self.sample_size = 0

        self.aug_config = aug_config

        positive_samples = []
        negative_samples = []

        for i, dname in enumerate(dataset_names):
            positive_path = os.path.join(dname, split_set, "Pos")
            prob = dataset_probs[i]

            ### samples positive samples
            samples = sorted(list(Path(positive_path).glob('[0-9]*')))
            prob_num = int(len(samples) * prob)
            samples = samples[:prob_num]
            print("Loading pos ", positive_path, len(samples))
            if split_set == "Val":
                samples = samples[:len(samples)//2]
            positive_samples.extend(samples)

            ### samples negative samples
            num_pos = len(samples)
            assert(neg_prob < 1)
            target_neg_num = int(num_pos/(1-neg_prob)*(neg_prob))

            negative_path = os.path.join(dname, split_set, "Neg")
            neg_samples = sorted(list(Path(negative_path).glob('[0-9]*')))
            print("Loading neg ", negative_path, len(neg_samples))
            if target_neg_num >= len(neg_samples):
                neg_samples = neg_samples
            else:
                neg_samples = neg_samples[:target_neg_num]
            negative_samples.extend(neg_samples)

        print(f"Positive {len(positive_samples)}, Negative {len(negative_samples)}")
        
        self.sample_size = 0
        self.datasets = {}
        self.datasets["positive"] = {
            "name": "positive", 
            "samples": positive_samples,
            "len": len(positive_samples),
            "start_index": self.sample_size,
            "end_index": self.sample_size + len(positive_samples),
        }
        self.sample_size += len(positive_samples)

        self.datasets["negative"] = {
            "name": "negative", 
            "samples": negative_samples,
            "len": len(negative_samples),
            "start_index": self.sample_size,
            "end_index": self.sample_size + len(negative_samples),
        }
        self.sample_size += len(negative_samples)


    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def tokenize_dialogue_label(self, dialogue):
        text = " ".join(dialogue.split('\n'))
        tokens = self.tokenizer.encode(text.strip(), return_tensors='pt')[0]
        return tokens
        
    # Dataset method - sample_size stores the sum of the lengths of all dataset sources
    def __len__(self):
        return self.sample_size

    # calculates the group an index falls into, and returns that group and the local index
    # returns tuple as dataset_name, index
    def shifted_index(self, i):
        # [start_index, end_index) - end is not inclusive
        for group in self.datasets.keys():
            if i < self.datasets[group]['end_index'] and i >= self.datasets[group]['start_index']:
                shifted_index = i - self.datasets[group]['start_index']
                return group, shifted_index
        return -1, -1
            
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        selected_dataset, shifted_i = self.shifted_index(i)
        sample = self.datasets[selected_dataset]['samples'][shifted_i]
        name = self.datasets[selected_dataset]['name']
        ### reading the dialogue, label and whisper from disk
        dialogue_text = ""
        whisper_text = ""

        if self.history_aware:
            dialogue_text = (Path(sample) / 'dialogue_aware.txt').read_text()
        else:
            dialogue_text = (Path(sample) / 'dialogue.txt').read_text()

        if name == "negative":
            whisper_text = ""
        else:
            whisper_text = (Path(sample) / 'whisper.txt').read_text()

        drop_men = np.random.rand() < self.mem_drop_rate
        if drop_men:
            memory_text = ""
        else:
            memory_text = (Path(sample) / 'memory.txt').read_text()

        ### add data augmenetation 
        # print("before", dialogue_text)
        if self.aug_config is not None:
            dialogue_text = augement_dialogue(dialogue_text, self.aug_config)
        # print("after", dialogue_text)
        ## prepare conversation template
        messages = [
            {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
            {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{memory_text}'},
            {"role": "user", "content": dialogue_text},
        ]
        conv_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_text = conv_text + whisper_text + '<|end_of_text|>'
        # print(all_text)
        ## The reason we use '<|end_of_text|>' 
        ## because the default data collator will replace all eos token to ignore_id
        ## so we first use '<|end_of_text|>' to avoid replacement and then in the data collator we replace it with eos id
        # tokenize the dialogue
        input_ids = self.tokenizer.encode(conv_text, return_tensors='pt')[0] #self.tokenize_dialogue_label(conv_text)
        label_ids = self.tokenizer.encode(all_text, return_tensors='pt')[0]
        padded_input_ids = torch.clone(label_ids) 

        if self.inference:
            # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)
            output = {
                "input_ids": padded_input_ids,
                "raw_input": input_ids,
                "raw_text": all_text
                # "labels": padded_label_ids,
            }
        else:   
            # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)
            ## the labels will be generated in data collator
            output = {
                "input_ids": padded_input_ids,
                # "raw_input": input_ids,
                # "raw_text": all_text
                # "labels": padded_label_ids,
            }
            
        return output
