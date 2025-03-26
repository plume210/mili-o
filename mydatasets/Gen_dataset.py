import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
import json
import os
import numpy as np 
from .data_augmentation import augement_dialogue


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
