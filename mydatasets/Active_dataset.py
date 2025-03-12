import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
import json
from .data_augmentation import augement_dialogue

import string

class Active_dataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        output_base: str,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train,
        negative_base: str = None, # NOTE this is for including negative samples, will fail if used and invalid
        # setting: string = None
    ):
        self.tokenizer = tokenizer
        self.datasets = {}
        self.quality_levels = {
            'claude': 3,
            'negative_claude': 3,
            # 'openai': 3,
            # 'seventyFull': 2,
            # 'seventyQuantized': 2,
            # 'eightB': 1
        }
        self.sample_size = 0

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = [f'{output_base}/{split_set}/{input_dir}' for input_dir in input_dirs]

        if negative_base is not None:
            dataset_lists.append(f'{negative_base}/{split_set}/negative_claude')

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder.split('/')[-1]
            print(f"Loading ... dataset: {dataset_folder}")
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            if dataset_name in self.quality_levels.keys():
                elements['quality'] = self.quality_levels[dataset_name]
            else:
                elements['quality'] = 1
            if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
                continue # skip it
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            
            elements['samples'] = samples
            print(len(elements['samples']))
            elements['len'] = len(elements['samples'])
            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1].split('/')[-1]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive

            # dataset_base = dataset_folder.split('/')[0]
            # if dataset_base == negative_base:
            #     self.datasets[f'{dataset_name}_negative'] = elements
            # else:
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
            
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        selected_dataset, shifted_i = self.shifted_index(i)
        sample = self.datasets[selected_dataset]['samples'][shifted_i]

        ### reading the dialogue, label and whisper from disk
        dialogue_text = ""
        label_text = ""
        answer_text = ""
        dialogue_text = (Path(sample) / 'dialogue.txt').read_text()
        label_text = (Path(sample) / 'values.txt').read_text()
        answer_text = (Path(sample) / 'whisper.txt').read_text()
        mask_text = (Path(sample) / 'mask.txt').read_text()
        

        # tokenize the dialogue
        dialogue = self.tokenize_dialogue_label(dialogue_text)
        # find the label - format as just a list of integers, then turn into a tensor
        label_joined = (" ".join(label_text.split('\n'))).split(' ')
        label_ints = list(map(int, label_joined))
        labels = torch.tensor(label_ints, dtype=torch.int)
        answer = self.tokenize_dialogue_label(answer_text)

        mask_joined = (" ".join(mask_text.split('\n'))).split(' ')
        mask_ints = list(map(int, mask_joined))
        mask = torch.tensor(mask_ints, dtype=torch.int)
        # print(dialogue.size, labels.size)
        # assert(dialogue.size() == labels.size())
        assert(mask.size() == labels.size())

        # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)
        # print(i, sample, dialogue.shape, labels.shape)

        ### merge mask and labels
        for i in range(labels.shape[-1]):
            if mask[i] == 0:
                labels[i] = -100
        example = {
            "input_ids": dialogue,
            "labels": labels,
            "answer_ids": answer,
            "mask": mask,
        }
        
        return example



class Active_WhisperAware_dataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        output_base: str,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train,
        negative_base: str = None, # NOTE this is for including negative samples, will fail if used and invalid
        # setting: string = None
    ):
        self.tokenizer = tokenizer
        self.datasets = {}
        self.quality_levels = {
            'claude': 3,
            'negative_claude': 3,
        }
        self.sample_size = 0

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = [f'{output_base}/{split_set}/{input_dir}' for input_dir in input_dirs]

        if negative_base is not None:
            dataset_lists.append(f'{negative_base}/{split_set}/negative_claude')

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder.split('/')[-1]
            print(f"Loading ... dataset: {dataset_folder}")
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            if dataset_name in self.quality_levels.keys():
                elements['quality'] = self.quality_levels[dataset_name]
            else:
                elements['quality'] = 1
            if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
                continue # skip it
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            
            elements['samples'] = samples
            print(len(elements['samples']))
            elements['len'] = len(elements['samples'])
            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1].split('/')[-1]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive

            # dataset_base = dataset_folder.split('/')[0]
            # if dataset_base == negative_base:
            #     self.datasets[f'{dataset_name}_negative'] = elements
            # else:
            self.datasets[dataset_name] = elements

    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def preprocess(self, text):
        text = " ".join(text.split('\n'))
        return text.strip()

        
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
            
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        Success = True
        bias = 0
        for bias in range(0, 5):
            Success = True
            selected_dataset, shifted_i = self.shifted_index(i + bias)
            sample = self.datasets[selected_dataset]['samples'][shifted_i]

            ### reading the dialogue, label and whisper from disk
            raw_text = (Path(sample) / 'raw.txt').read_text()
            answer_text = (Path(sample) / 'whisper.txt').read_text()
            raw_text = self.preprocess(raw_text)
            replys = answer_text.splitlines()
            reply_num = raw_text.count(" ^^")

            if (reply_num != len(replys)):
                Success = False
                continue

            replacements = iter(replys)
            raw_text = raw_text.replace(" ^^", " ^^ Agent: {}").format(*replacements)
            raw_diag = raw_text.replace(" ^^", "")

            whisper_token = self.tokenizer(" ^^")['input_ids'][-1]
            symbol_token = self.tokenizer(" |")['input_ids'][-1]
            symbol_token2 = self.tokenizer(" >")['input_ids'][-1]

            raw_tokens = self.tokenizer.encode(raw_text, return_tensors='pt')[0]
            dialogue = self.tokenizer.encode(raw_diag, return_tensors='pt')[0]

            labels = []
            masks = []
            
            for _i in range(raw_tokens.shape[-1]):
                token = raw_tokens[_i].item()
                if token == whisper_token:
                    labels[-1] = 1
                    masks[-1] = 1
                else:
                    labels.append(0)
                    if token == symbol_token2:
                        masks.append(1)
                    else:
                        masks.append(0)

            masks = torch.tensor(masks, dtype=torch.int)
            labels = torch.tensor(labels, dtype=torch.int)
            assert(masks.size() == masks.size())
            assert(masks.size() == dialogue.size())

            ### merge mask and labels
            for i in range(labels.shape[-1]):
                if labels[i] == 1:
                    if dialogue[i] != symbol_token2 :
                        Success = False
                        break
                    # assert(dialogue[i] == symbol_token2)
                if masks[i] == 0:
                    labels[i] = -100

            if Success == False:
                continue

            example = {
                "input_ids": dialogue,
                "labels": labels,
                "mask": masks,
            }
            
            return example

        raise ValueError("Some bugs happens in dataset !!!!!!")


class New_WhisperAware_dataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train,
        negative_base: str = None, # NOTE this is for including negative samples, will fail if used and invalid
        # setting: string = None
        aug_config: dict = None
    ):
        self.tokenizer = tokenizer
        self.datasets = {}
        self.sample_size = 0
        self.aug_config = aug_config

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = input_dirs

        if negative_base is not None:
            dataset_lists.append(negative_base)

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder
            print(f"Loading ... dataset: {dataset_folder}")
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            elements['quality'] = 1

            if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
                continue # skip it
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            if "active_agent_fix" in dataset_name:
                samples = samples[:len(samples)//3]
            elements['samples'] = samples
            elements['len'] = len(elements['samples'])
            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive

            # dataset_base = dataset_folder.split('/')[0]
            # if dataset_base == negative_base:
            #     self.datasets[f'{dataset_name}_negative'] = elements
            # else:
            self.datasets[dataset_name] = elements

    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def preprocess(self, text):
        text = " ".join(text.split('\n'))
        return text.strip()

        
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
        raise ValueError("Index is wrong or not found in datasets keys!!!", i)
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        Success = True
        bias = 0
        for bias in range(0, 5):
            # print("!!!!!!!!!!!!", bias)
            Success = True
            selected_dataset, shifted_i = self.shifted_index(i + bias)
            sample = self.datasets[selected_dataset]['samples'][shifted_i]
            ### reading the dialogue, label and whisper from disk
            raw_text = (Path(sample) / 'raw.txt').read_text()
            answer_text = (Path(sample) / 'whisper.txt').read_text()
            raw_text = self.preprocess(raw_text)
            raw_text0 = raw_text
            ## add data augment 
            
            if self.aug_config is not None:
                # print("before", raw_text)
                raw_text = augement_dialogue(raw_text, self.aug_config)
                # print("----")
                # print("after", raw_text)

            replys = answer_text.splitlines()
            reply_num = raw_text.count(" ^^")

            if (reply_num != len(replys)):
                print("Warining!!!!! reply times not equal to reply lines")
                Success = False
                continue

            replacements = iter(replys)
            if self.aug_config is not None:
                raw_text = raw_text.replace(" ^^", " ^^ agent: {}").format(*replacements)
            else:
                raw_text = raw_text.replace(" ^^", " ^^ Agent: {}").format(*replacements)
            raw_diag = raw_text.replace(" ^^", "")

            whisper_token = self.tokenizer(" ^^")['input_ids'][-1]
            # symbol_token = self.tokenizer(" |")['input_ids'][-1]
            symbol_token2 = self.tokenizer(" >")['input_ids'][-1]

            raw_tokens = self.tokenizer.encode(raw_text, return_tensors='pt')[0]
            dialogue = self.tokenizer.encode(raw_diag, return_tensors='pt')[0]

            labels = []
            masks = []
            for _i in range(raw_tokens.shape[-1]):
                token = raw_tokens[_i].item()
                # print(raw_tokens[_i], dialogue[_i])
                if token == whisper_token:
                    labels[-1] = 1
                    masks[-1] = 1
                else:
                    labels.append(0)
                    if token == symbol_token2:
                        masks.append(1)
                    else:
                        masks.append(0)

            masks = torch.tensor(masks, dtype=torch.int)
            labels = torch.tensor(labels, dtype=torch.int)
            assert(masks.size() == labels.size())
            assert(masks.size() == dialogue.size())

            ### merge mask and labels
            for i in range(labels.shape[-1]):
                if labels[i] == 1:
                    if dialogue[i] != symbol_token2 :
                        Success = False
                        # print("Warning!!! label and mask are not consistent ", sample)
                        # print(raw_text0)
                        # print("------")
                        # print(raw_text)
                        # exit(0)
                        break
                    # assert(dialogue[i] == symbol_token2)
                if masks[i] == 0:
                    labels[i] = -100

            if Success == False:
                continue


            # for i in range(labels.shape[-1]):
            #     # if labels[i] != -100:
            #     print(labels[i], masks[i], dialogue[i], raw_tokens[i])
            example = {
                "input_ids": dialogue,
                "labels": labels,
                "mask": masks,
            }
            
            return example

        raise ValueError("Some bugs happens in dataset !!!!!!")




class New_WhisperAware_dataset2(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train,
        negative_base: str = None, # NOTE this is for including negative samples, will fail if used and invalid
        history_aware = True,
        # setting: string = None
    ):
        self.tokenizer = tokenizer
        self.history_aware = history_aware
        self.datasets = {}
        self.sample_size = 0

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = input_dirs

        if negative_base is not None:
            dataset_lists.append(negative_base)

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            elements['quality'] = 1

            if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
                continue # skip it
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            if "active_agent_fix" in dataset_name:
                samples = samples[:len(samples)//2]
            elements['samples'] = samples
            elements['len'] = len(elements['samples'])

            print(f"Loading ... dataset: {dataset_folder}, sample size = {len(samples)}")

            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive

            # dataset_base = dataset_folder.split('/')[0]
            # if dataset_base == negative_base:
            #     self.datasets[f'{dataset_name}_negative'] = elements
            # else:
            self.datasets[dataset_name] = elements

    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def preprocess(self, text):
        text = " ".join(text.split('\n'))
        return text.strip()

        
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
        raise ValueError("Index is wrong or not found in datasets keys!!!", i)
    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        Success = True
        bias = 0
        for bias in range(0, 5):
            Success = True
            selected_dataset, shifted_i = self.shifted_index(i + bias)
            sample = self.datasets[selected_dataset]['samples'][shifted_i]

            ### reading the dialogue, label and whisper from disk
            raw_text = (Path(sample) / 'raw.txt').read_text()
            answer_text = (Path(sample) / 'whisper.txt').read_text()
            raw_text = self.preprocess(raw_text)

            replys = answer_text.splitlines()
            reply_num = raw_text.count(" ^^")

            if (reply_num != len(replys)):
                print("Warining!!!!! reply times not equal to reply lines")
                Success = False
                continue

            replacements = iter(replys)
            if self.history_aware:
                raw_text = raw_text.replace(" ^^", " ^^ Agent: {}").format(*replacements)
            raw_diag = raw_text.replace(" ^^", "")

            whisper_token = self.tokenizer(" ^^")['input_ids'][-1]
            symbol_token = self.tokenizer(" |")['input_ids'][-1]
            symbol_token2 = self.tokenizer(" >")['input_ids'][-1]

            raw_tokens = self.tokenizer.encode(raw_text, return_tensors='pt')[0]
            dialogue = self.tokenizer.encode(raw_diag, return_tensors='pt')[0]

            labels = []
            masks = []
            
            for _i in range(raw_tokens.shape[-1]):
                token = raw_tokens[_i].item()
                if token == whisper_token:
                    labels[-1] = 1
                    masks[-1] = 1
                else:
                    labels.append(0)
                    if token == symbol_token2:
                        masks.append(1)
                    else:
                        masks.append(0)

            masks = torch.tensor(masks, dtype=torch.int)
            labels = torch.tensor(labels, dtype=torch.int)
            assert(masks.size() == masks.size())
            assert(masks.size() == dialogue.size())

            ### merge mask and labels
            for i in range(labels.shape[-1]):
                if labels[i] == 1:
                    if dialogue[i] != symbol_token2 :
                        Success = False
                        break
                    # assert(dialogue[i] == symbol_token2)
                if masks[i] == 0:
                    labels[i] = -100

            if Success == False:
                continue

            example = {
                "input_ids": dialogue,
                "labels": labels,
                "mask": masks,
            }
            
            return example

        raise ValueError("Some bugs happens in dataset !!!!!!")


class New_Active_dataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        input_dirs: list,
        split_set: str = "Train",
        quality_filter: int = 0, # 0 for all qualities, should only use for train - will ignore if not train,
        negative_base: str = None, # NOTE this is for including negative samples, will fail if used and invalid
        # setting: string = None
    ):
        self.tokenizer = tokenizer
        self.datasets = {}
        self.sample_size = 0

        # add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_lists = input_dirs

        if negative_base is not None:
            dataset_lists.append(negative_base)

        # load the samples
        for dataset_idx, dataset_folder in enumerate(dataset_lists):
            dataset_name = dataset_folder
            
            elements = {'name': "", 'quality': -1, 'samples': [], 'len': -1, 'end_index': -1, 'start_index': 0}
            elements['name'] = dataset_name
            elements['quality'] = 1

            if split_set == "Train" and quality_filter > 0 and quality_filter != elements['quality']:
                continue # skip it
            samples = sorted(list(Path(dataset_folder).glob('[0-9]*')))
            if "active_agent_fix" in dataset_name:
                samples = samples[:int(len(samples)*0.4)]
            
            print(f"Loading ... dataset: {dataset_folder}, sample size = {len(samples)}")

            elements['samples'] = samples
            elements['len'] = len(elements['samples'])
            self.sample_size += elements['len']

            # calculate "start" and "end" indices
            #   - if all datasets were lined up, calculates what the indices would be
            elements['end_index'] = len(elements['samples'])
            if dataset_idx > 0:
                prev_name = dataset_lists[dataset_idx - 1]
                elements['start_index'] += self.datasets[prev_name]['len'] + self.datasets[prev_name]['start_index']
                elements['end_index'] += elements['start_index'] # not inclusive

            # dataset_base = dataset_folder.split('/')[0]
            # if dataset_base == negative_base:
            #     self.datasets[f'{dataset_name}_negative'] = elements
            # else:
            self.datasets[dataset_name] = elements

    # tokenizes a string - joins all lines if they aren't joined already
    # returns as a pytorch tensor
    def preprocess(self, text):
        text = " ".join(text.split('\n'))
        return text.strip()

        
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
        raise ValueError("Index is wrong or not found in datasets keys!!!", i)

    def tokenize_dialogue_label(self, dialogue):
        text = " ".join(dialogue.split('\n'))
        tokens = self.tokenizer.encode(text.strip(), return_tensors='pt')[0]
        return tokens

    # dataset method - to do something like `dataset[i]`
    def __getitem__(self, i):
        # get shifted index and sample
        selected_dataset, shifted_i = self.shifted_index(i)
        sample = self.datasets[selected_dataset]['samples'][shifted_i]

        ### reading the dialogue, label and whisper from disk
        dialogue_text = ""
        label_text = ""
        answer_text = ""
        dialogue_text = (Path(sample) / 'dialogue.txt').read_text()
        label_text = (Path(sample) / 'values.txt').read_text()
        answer_text = (Path(sample) / 'whisper.txt').read_text()
        mask_text = (Path(sample) / 'mask.txt').read_text()
        

        # tokenize the dialogue
        dialogue = self.tokenize_dialogue_label(dialogue_text)
        # find the label - format as just a list of integers, then turn into a tensor
        label_joined = (" ".join(label_text.split('\n'))).split(' ')
        label_ints = list(map(int, label_joined))
        labels = torch.tensor(label_ints, dtype=torch.int)
        answer = self.tokenize_dialogue_label(answer_text)

        mask_joined = (" ".join(mask_text.split('\n'))).split(' ')
        mask_ints = list(map(int, mask_joined))
        mask = torch.tensor(mask_ints, dtype=torch.int)
        # print(dialogue.size, labels.size)
        # assert(dialogue.size() == labels.size())
        assert(mask.size() == labels.size())

        # returns the dialogue (without agent) (as token tensor), the labels (as tensor), and the whisper response (as token tensor)

        ### merge mask and labels
        for i in range(labels.shape[-1]):
            if mask[i] == 0:
                labels[i] = -100
        example = {
            "input_ids": dialogue,
            "labels": labels,
            "answer_ids": answer,
            "mask": mask,
        }
        
        return example
