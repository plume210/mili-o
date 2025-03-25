import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
import json
import csv
import os
import re

import string
START_INDEX = 0

class Syn_samples(object):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        gen_tokenizer: PreTrainedTokenizer,
        output_base: str,
        sample_id: int,
        split_set: str = "Test",
        input_dirs: list = None
    ): 
        self.tokenizer = tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.datasets = {}
        self.sample_size = 0

        samples = sorted(list(Path(output_base).glob('[0-9]*')))    
        self.sample_size += len(samples)

        self.valid = True
        if sample_id >= self.sample_size:
            self.valid = False
            return 

        self.dataset = samples
        sample = self.dataset[sample_id]
        ### reading the dialogue, label and whisper from disk
        self.dialogue_text = ""
        self.label_text = ""
        self.answer_text = ""
        self.memory_text = ""
        self.dialogue_text = (Path(sample) / 'dialogue.txt').read_text()
        self.label_text = (Path(sample) / 'values.txt').read_text()
        self.answer_text = (Path(sample) / 'whisper.txt').read_text()
        self.mask_text = (Path(sample) / 'mask.txt').read_text()
        self.memory_text = (Path(sample) / 'memory.txt').read_text()
        self.answer_list = self.answer_text.split('\n')

        # create stream for speech 
        self.tokenized_dialogue = self.tokenizer.encode(self.dialogue_text, return_tensors='pt')[0]
        self.tokenized_dialogue_history = self.tokenized_dialogue.clone()

        self.stream_id = START_INDEX
        self.stream_id_history = START_INDEX
        
        # gt label
        label_joined = (" ".join(self.label_text.split('\n'))).split(' ')
        label_ints = list(map(int, label_joined))
        self.labels = torch.tensor(label_ints, dtype=torch.int)

        mask_joined = (" ".join(self.mask_text.split('\n'))).split(' ')
        mask_ints = list(map(int, mask_joined))
        new_mask = [0 for i in mask_ints]
        for i in range(len(mask_ints)):
            if mask_ints[i]:
                new_mask[i] = 1
        #         if i > 1:
        #             new_mask[i - 1] = 1
        #         if  i + 1 < len(mask_ints):
        #             new_mask[i + 1] = 1

        # print("Answer!!!!!", self.answer_list)
        self.mask = torch.tensor(mask_ints, dtype=torch.int)
        self.mask_history = self.mask.clone()
        assert(self.tokenized_dialogue.size() == self.mask.size())
        assert(self.tokenized_dialogue_history.size() == self.mask_history.size())
        assert(self.mask.size() == self.labels.size())

    def get_mem(self):
        return self.memory_text

    def count_turn(self):
        # return self.dialogue_text.count("User:") + self.dialogue_text.count("Speaker 1:") + self.dialogue_text.count("Speaker1:")
        return self.dialogue_text.count("User:") + sum([self.dialogue_text.count(f"Speaker {n}:") + self.dialogue_text.count(f"Speaker{n}:") for n in range(1,6)])

    def reset_streaming(self):
        self.stream_id = START_INDEX 
        self.stream_id_history = START_INDEX

    def insert_whisper(self, whisper_text):
        whisper_token = self.tokenizer.encode(whisper_text, return_tensors='pt', add_special_tokens = False)[0]
        L_whisper = whisper_token.shape[-1]
        whisper_mask = torch.zeros_like(whisper_token)

        self.tokenized_dialogue_history = torch.cat([self.tokenized_dialogue_history[:self.stream_id_history], whisper_token, self.tokenized_dialogue_history[self.stream_id_history:]]  )
        self.mask_history = torch.cat([self.mask_history[:self.stream_id_history], whisper_mask, self.mask_history[self.stream_id_history:]]  )

        assert(self.tokenized_dialogue_history.size() == self.mask_history.size())

        self.stream_id_history += L_whisper


    def streaming_diaglogue(self):
        if self.stream_id < len(self.tokenized_dialogue) and self.stream_id_history < len(self.tokenized_dialogue_history):
            curr_token = self.tokenized_dialogue[:1+self.stream_id]
            curr_token_history = self.tokenized_dialogue_history[:1+self.stream_id_history]
            curr_mask = self.mask[:1+self.stream_id]
            curr_mask_history = self.mask_history[:1+self.stream_id_history]
            curr_label = self.labels[:1+self.stream_id]
            self.stream_id += 1
            self.stream_id_history += 1
            return curr_token, curr_token_history, curr_mask, curr_mask_history, curr_label
        else:
            return None 
        

    def snap_dialogue(self):
        return self.tokenized_dialogue, self.tokenized_dialogue_history
    
    
    def get_gen_inputs(self, curr_diag, old=False):
        ## prepare conversation template
        messages = [
            {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
            {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{self.memory_text}'},
            {"role": "user", "content": curr_diag},
        ]
        if old:
            messages = [
                {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3) words to its user. You will be presented with a set of rules and a dialogue."},
                {'role': 'user', 'content': 'We define nine principles to guide desired proactive agent behavior. \n- Valuable: advances the user’s interests and tasks, in the user’s opinion \n- Pertinent: attentive to the current situation \n- Competent: within the scope of the agent’s abilities and knowledge \n- Unobtrusive: not interfering with the user’s own activities or attention, without warrant - Transparent: understandable to the user \n- Controllable: exposed to the scrutiny and according to the mandate of the user \n- Deferent: gracefully unimposing \n- Anticipatory: aware of current and future needs and opportunities \n- Safe: minimizes negative consequences, in the user’s opinion \nThe user has three use cases for a proactive agent: \n1. Reminding. \n2. Social Guidance: Scenarios that warrant social guidance may involve an interview, first date, or public speaking, etc. \n3. Managing emotional dysregulation. \n \nHere are some guidelines for answering questions: \n1. Your answer must be at most 3 English words long. After your answer is outputted, stop generating. \n2. Your answer must be in accordance with the nine principles for proactive agents. \n3. Your answer must not contain any emotion tokens, or any of "|" \n4. Do not generate anything except for your answer. This includes any notes in parentheses, or explanations. \n5. You are an audio assistant, and you are speaking directly to the user through audio. Only output the answer, nothing else.'},
                {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{self.memory_text}'},
                {"role": "user", "content": f'Dialogue:\n{curr_diag}'},
                {"role": 'user', 'content': 'What is the best Answer in accordance with the rules?'}
            ]
        conv_text = self.gen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # tokenize the dialogue
        input_ids = self.gen_tokenizer.encode(conv_text, return_tensors='pt')[0] #self.tokenize_dialogue_label(conv_text)

        return input_ids


class MIT_sample(object):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        gen_tokenizer: PreTrainedTokenizer,
        output_base: str,
        sample_id: int,
        split_set: str = "Train",
        input_dirs: list = None
    ):
        self.tokenizer = tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.datasets = {}
        self.sample_size = 0

        dataset_lists = output_base
        # load the samples
        dataset_folder = dataset_lists
        # print(f"Loading ... dataset: {dataset_folder}")

        samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))          
        # print(len(samples))
        self.sample_size += len(samples)
        self.dataset = samples
        self.valid = True
        if sample_id >= self.sample_size:
            self.valid = False
            return 
        # extract sample information
        sample = self.dataset[sample_id]
        ### reading the dialogue, label and whisper from disk
        self.dialogue_text = ""
        self.memory_text = ""
        self.mask_text = ""
        self.dialogue_text = (Path(sample) / 'dialogue.txt').read_text()
        self.mask_text = (Path(sample) / 'mask.txt').read_text()
        # self.memory_text = (Path(sample) / 'memory.txt').read_text()
        # create stream for speech 
        self.tokenized_dialogue = self.tokenizer.encode(self.dialogue_text, return_tensors='pt')[0]
        self.tokenized_dialogue_history = self.tokenized_dialogue.clone()

        self.stream_id = START_INDEX
        self.stream_id_history = START_INDEX

        mask_joined = (" ".join(self.mask_text.split('\n'))).split(' ')
        mask_ints = list(map(int, mask_joined))
        new_mask = [0 for i in mask_ints]
        for i in range(len(mask_ints)):
            if mask_ints[i]:
                new_mask[i] = 1
        #         if i > 1:
        #             new_mask[i - 1] = 1
        #         if  i + 1 < len(mask_ints):
        #             new_mask[i + 1] = 1


        self.mask = torch.tensor(mask_ints, dtype=torch.int)
        self.mask_history = self.mask.clone()
        # print(self.tokenized_dialogue.size() , self.mask.size() ,  self.labels.size())
        # print(self.tokenized_dialogue.size() , self.mask.size())
        assert(self.tokenized_dialogue.size() == self.mask.size())
        assert(self.tokenized_dialogue_history.size() == self.mask_history.size())

    def count_turn(self):
        # return self.dialogue_text.count("User:") + self.dialogue_text.count("Speaker 1:") + self.dialogue_text.count("Speaker1:")
        return self.dialogue_text.count("User:") + sum([self.dialogue_text.count(f"Speaker {n}:") + self.dialogue_text.count(f"Speaker{n}:") for n in range(1,6)])
    
    def reset_streaming(self):
        self.stream_id = START_INDEX 
        self.stream_id_history = START_INDEX

    def insert_whisper(self, whisper_text):
        whisper_token = self.tokenizer.encode(whisper_text, return_tensors='pt', add_special_tokens = False)[0]
        L_whisper = whisper_token.shape[-1]
        whisper_mask = torch.zeros_like(whisper_token)

        self.tokenized_dialogue_history = torch.cat([self.tokenized_dialogue_history[:self.stream_id_history], whisper_token, self.tokenized_dialogue_history[self.stream_id_history:]]  )
        self.mask_history = torch.cat([self.mask_history[:self.stream_id_history], whisper_mask, self.mask_history[self.stream_id_history:]]  )

        assert(self.tokenized_dialogue_history.size() == self.mask_history.size())

        self.stream_id_history += L_whisper



    def streaming_diaglogue(self):
        if self.stream_id < len(self.tokenized_dialogue) and self.stream_id_history < len(self.tokenized_dialogue_history):
            curr_token = self.tokenized_dialogue[:1+self.stream_id]
            curr_token_history = self.tokenized_dialogue_history[:1+self.stream_id_history]
            curr_mask = self.mask[:1+self.stream_id]
            curr_mask_history = self.mask_history[:1+self.stream_id_history]
            self.stream_id += 1
            self.stream_id_history += 1
            return curr_token, curr_token_history, curr_mask, curr_mask_history
        else:
            return None 
        

    def snap_dialogue(self):
        return self.tokenized_dialogue, self.tokenized_dialogue_history
    
    
    def get_gen_inputs(self, curr_diag, old=False):
        ## prepare conversation template
        messages = [
            {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
            {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{self.memory_text}'},
            {"role": "user", "content": curr_diag},
        ]
        if old:
            messages = [
                {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3) words to its user. You will be presented with a set of rules and a dialogue."},
                {'role': 'user', 'content': 'We define nine principles to guide desired proactive agent behavior. \n- Valuable: advances the user’s interests and tasks, in the user’s opinion \n- Pertinent: attentive to the current situation \n- Competent: within the scope of the agent’s abilities and knowledge \n- Unobtrusive: not interfering with the user’s own activities or attention, without warrant - Transparent: understandable to the user \n- Controllable: exposed to the scrutiny and according to the mandate of the user \n- Deferent: gracefully unimposing \n- Anticipatory: aware of current and future needs and opportunities \n- Safe: minimizes negative consequences, in the user’s opinion \nThe user has three use cases for a proactive agent: \n1. Reminding. \n2. Social Guidance: Scenarios that warrant social guidance may involve an interview, first date, or public speaking, etc. \n3. Managing emotional dysregulation. \n \nHere are some guidelines for answering questions: \n1. Your answer must be at most 3 English words long. After your answer is outputted, stop generating. \n2. Your answer must be in accordance with the nine principles for proactive agents. \n3. Your answer must not contain any emotion tokens, or any of "|" \n4. Do not generate anything except for your answer. This includes any notes in parentheses, or explanations. \n5. You are an audio assistant, and you are speaking directly to the user through audio. Only output the answer, nothing else.'},
                {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{self.memory_text}'},
                {"role": "user", "content": f'Dialogue:\n{curr_diag}'},
                {"role": 'user', 'content': 'What is the best Answer in accordance with the rules?'}
            ]
        # if old:
        conv_text = self.gen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # tokenize the dialogue
        # print(conv_text)
        input_ids = self.gen_tokenizer.encode(conv_text, return_tensors='pt')[0] #self.tokenize_dialogue_label(conv_text)

        return input_ids


