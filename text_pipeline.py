from datasets import load_dataset
import datasets
from pathlib import Path
import re
import numpy as np
import os
from peft import LoraConfig, get_peft_model, TaskType
os.environ["WANDB_DISABLED"] = "true"
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from model.CasualTokenClassificationLlama import LlamaForCausalLM_TokenClassifcation

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score
from peft import PeftModel, PeftConfig
import json



def find_best_ckpt(root_folder, eval_metric = "eval_loss"):
    # List all subfolders that match the "checkpoint-xxx" pattern
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    folders = []

    # Iterate through all subfolders
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            match = checkpoint_pattern.match(subfolder)
            if match:
                folders.append((subfolder, int(match.group(1))))

    # Find the folder with the largest xxx
    if folders:
        largest_checkpoint = max(folders, key=lambda x: x[1])
        print(f"Folder with the largest ckpt: {largest_checkpoint[0]}")
        latest =  os.path.join(root_folder, largest_checkpoint[0])
    else:
        print("No checkpoint folders found.")
        return None
    
    with open(os.path.join(latest, "trainer_state.json"), "r", encoding="utf-8") as file:
        data = json.load(file)

    logging = data["log_history"]
    val_logging = logging[1::2]
    steps = []
    vals = []
    for log in val_logging:
        steps.append(log["step"])
        vals.append(log[eval_metric])
    if eval_metric == "eval_loss":
        best_id = np.argmin(vals)
        best_step = steps[best_id]
        best_ckpt = os.path.join(root_folder, f"checkpoint-{best_step}")
        print("Find best ckpt ---- ", best_ckpt)
        return  f"checkpoint-{best_step}"
    elif eval_metric == "eval_accuracy":
        best_id = np.argmax(vals)
        best_step = steps[best_id]
        best_ckpt = os.path.join(root_folder, f"checkpoint-{best_step}")
        print("Find best ckpt ---- ", best_ckpt)
        return  f"checkpoint-{best_step}"
    else:
        raise ValueError(f"{eval_metric} metrics not found ")

classfier = "neg_4data" # basic_4data, neg_4data, whisper_aware_4data
generator = "pos_4data" # pos_4data, lowneg_4data silent_aware_4data, lowneg_4data_1b
dataset_name = "MIT" 

if dataset_name in ["MIT"]:
    output_samples = f"/gscratch/intelligentsystems/tuochao/Proactive_Agent/real_result/{dataset_name}_{classfier}_{generator}"
else:
    output_samples = f"/gscratch/intelligentsystems/tuochao/Proactive_Agent/syn_result/{dataset_name}_{classfier}_{generator}"
# output_base = '/gscratch/intelligentsystems/common_datasets/active_agent/Reformatted/Generation/'
os.makedirs(output_samples, exist_ok = True)

if dataset_name == "MIT":
    from mydatasets.Pipeline_dataset import MIT_sample  as  SingleSample
    output_base = '/gscratch/intelligentsystems/common_datasets/active_agent/MIT_final/'
    input_dirs = []
elif dataset_name == "Sync_claude":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/synthetic/Test/claude"
    # "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/perl/Test/claude"
    input_dirs = []
elif dataset_name == "Sync_perl":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/perl/Test/claude"
    input_dirs = []
elif dataset_name == "Sync_soda":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/soda/Test/claude"
    # "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/perl/Test/claude"
    input_dirs = []
else:
    raise ValueError("dataset not supported!") 


classifier_aware = False
generator_aware = False

if classfier == "basic_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_basic_fix/"
    best_ckpt = find_best_ckpt(ckpt_path, eval_metric = "eval_accuracy")
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    class_model_arg = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
        "ckpt_path": ckpt_path,
        "ckpt_num": best_ckpt
    }
    active_factor = 0
    classifier_aware = False

elif classfier == "neg_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_neg_fix/"
    best_ckpt = find_best_ckpt(ckpt_path, eval_metric = "eval_accuracy")
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    class_model_arg = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
        "ckpt_path": ckpt_path,
        "ckpt_num": best_ckpt
    }
    active_factor = 0
    classifier_aware = False

elif classfier == "whisper_aware_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_whisper_aware/"
    best_ckpt = find_best_ckpt(ckpt_path, eval_metric = "eval_accuracy")
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    class_model_arg = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
        "ckpt_path": ckpt_path,
        "ckpt_num": best_ckpt
    }
    active_factor = 0
    classifier_aware = True
else:
    raise ValueError("classifier not supported!")

if generator == "pos_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_4data_pos/"
    best_ckpt = find_best_ckpt(ckpt_path)
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    gen_model_arg = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_1",
        "ckpt_path":  ckpt_path,
        "ckpt_num":best_ckpt
    }
    generator_aware = True
elif generator == "lowneg_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_4data_lowneg/"
    best_ckpt = find_best_ckpt(ckpt_path)
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    gen_model_arg = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_1",
        "ckpt_path":  ckpt_path,
        "ckpt_num":best_ckpt
    }
    generator_aware = True
elif generator == "silent_aware_4data":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_4data/"
    best_ckpt = find_best_ckpt(ckpt_path)
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    gen_model_arg = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_1",
        "ckpt_path":  ckpt_path,
        "ckpt_num":best_ckpt
    }
    generator_aware = True
elif generator == "lowneg_4data_1b":
    ckpt_path = "/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_4data_1b/"
    best_ckpt = find_best_ckpt(ckpt_path)
    if best_ckpt is None:
        raise ValueError("ckpt does not exists!!!!")

    gen_model_arg = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_cache": "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
        "ckpt_path":  ckpt_path,
        "ckpt_num":best_ckpt
    }
    generator_aware = True
else:
    raise ValueError("classifier not supported!")  

### loading the classifier model
tokenizer_small = AutoTokenizer.from_pretrained(
    class_model_arg["model_id"],
    cache_dir = class_model_arg["model_cache"],
    pad_token="<|eot_id|>",
    token = "xxx"
    )

tokenizer_small.pad_token = tokenizer_small.eos_token
model_small = LlamaForCausalLM_TokenClassifcation.from_pretrained(
    class_model_arg["model_id"],
    device_map='cuda', 
    torch_dtype=torch.bfloat16, 
    num_labels = 2,
    cache_dir = class_model_arg["model_cache"],
    token = "xxx")

config = PeftConfig.from_pretrained(class_model_arg["ckpt_path"] + class_model_arg["ckpt_num"])
model_small = PeftModel.from_pretrained(model_small, class_model_arg["ckpt_path"] + class_model_arg["ckpt_num"], is_trainable=True) # 👈 here,

# model_small = model_small.merge_and_unload()
# print(model_small)
# model_small.push_to_hub("tuochao/Llama-3.2-1B-Proactive-Classifier-Aug", token="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")
# tokenizer_small.push_to_hub("tuochao/Llama-3.2-1B-Proactive-Classifier-Aug", token="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")
# exit(0)

"""
add to hg config 
"num_labels": 2, 
"model_type": "llama_classifier",
"""


### load the gen model
tokenizer_big = AutoTokenizer.from_pretrained(
    gen_model_arg["model_id"],
    cache_dir = gen_model_arg["model_cache"],
    pad_token="<|eot_id|>",
    token = "xxxx"
    )

tokenizer_big.pad_token = tokenizer_big.eos_token
model_big = AutoModelForCausalLM.from_pretrained(
    gen_model_arg["model_id"],
    device_map='cuda', 
    torch_dtype=torch.bfloat16, 
    cache_dir = gen_model_arg["model_cache"],
    token = "xxx")

config = PeftConfig.from_pretrained(gen_model_arg["ckpt_path"] + gen_model_arg["ckpt_num"])
model_big = PeftModel.from_pretrained(model_big, gen_model_arg["ckpt_path"] + gen_model_arg["ckpt_num"], is_trainable=True) # 👈 here,
# model_big = model_big.merge_and_unload()
### load dataset 
# model_big.push_to_hub("tuochao/Llama-3.1-8B-Proactive-Gen-Positive", token="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")
# tokenizer_big.push_to_hub("tuochao/Llama-3.1-8B-Proactive-Gen-Positive", token="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")


terminators = [
    tokenizer_big.eos_token_id,
    tokenizer_big.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer_big.convert_tokens_to_ids("<|end_of_text|>")
]
response_template = "<|start_header_id|>assistant<|end_header_id|>"

FN = 0
FP = 0
TP = 0
TN = 0

Num_turns = 0
Num_trigger_class = 0
Num_trigger_gen = 0
diag_id = 0

for i in range(0, 100):
    print("----------", i)
    torch.cuda.empty_cache()
    sample = SingleSample(tokenizer_small, tokenizer_big, output_base, split_set="Test", sample_id = i, input_dirs = input_dirs)
    
    if not sample.valid:
        continue

    Num_turns += sample.count_turn()
    L_cache = 0
    save_id = 0
    previous_id = 0

    save_dir = output_samples  + "/{:05d}".format(i)
    # os.makedirs(save_dir, exist_ok = True)
    curr_turn = ""
    curr_response = ""
    while True:
        info = sample.streaming_diaglogue()
        if info is None:
            break
        
        if "Sync" in dataset_name:
            token, token_history, mask, mask_history, label = info 
            curr_label = label[-1]
        else:
            token, token_history, mask, mask_history = info 

        ### check whether the classifier model is history aware
        if classifier_aware:
            token_small = token_history
            mask_small = mask_history
        else:
            token_small = token
            mask_small = mask

        # print("mask", mask[-1])
        if mask_small[-1] == 0:
            L_cache += 1
            continue # skip the non-special token 

        ### start infer on small model
        intput_ids = token_small.unsqueeze(0).to(model_small.device)
        out1 = model_small(input_ids = intput_ids)
        pred = out1.logits[0].cpu()
        pred[..., 1] += active_factor # make it more proactive
        # print(intput_ids)
        # print(active_factor, pred)
        pred = torch.argmax(pred, dim=-1)
        
        # exit(0)

        new_pred = pred[-1]
        new_mask = mask_small[-1]
        assert(new_mask == 1)
        assert(mask_history[-1] == 1)
        assert(mask[-1] == 1)

        ### start infer on large model
        if generator_aware:
            token_big = token_history
            mask_big = mask_history
        else:
            token_big = token
            mask_big = mask
        
        if "Sync" in dataset_name:
            if new_pred == 1 and curr_label == 1:
                TP += 1
            elif new_pred == 1 and curr_label == 0:
                FP += 1
            elif new_pred == 0 and curr_label == 1:
                FN += 1
            else:
                TN += 1

        if new_pred == 1:
            print("------------- agent is triggered ----------------")
            Num_trigger_class += 1

            curr_diag = tokenizer_small.decode(token_big, skip_special_tokens = True)

            with open(f'./debug/diag{diag_id}.txt', 'w') as f:
                f.write(curr_diag)
            diag_id += 1

            input_ids2 = sample.get_gen_inputs(curr_diag)
            input_ids2 = input_ids2.unsqueeze(0).to(model_big.device)
            outputs = model_big.generate(
                input_ids2,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.6,
                top_p=0.9,
                pad_token_id = tokenizer_big.pad_token_id
            )
            response = outputs[0]#[input_ids.shape[-1]:]
            output = tokenizer_big.decode(response, skip_special_tokens=False)
            output = output.split(response_template)
            raw_diag = output[0]
            response = output[1]
            response = response.replace("<|eot_id|>", "")
            response = response.replace("\n", "")

            ### add whisper back if whisper_aware
            if  len(response) >= 1:
                curr_response = response # model did wshiper sth
                print(curr_diag)
                print("***** Agent *****: ", response)
                Num_trigger_gen += 1
                sample.insert_whisper(" Agent: " + response)
        L_cache += 1

    tokens, tokens_history = sample.snap_dialogue()
    diag = tokenizer_small.decode(tokens_history, skip_special_tokens = True)

    data = {
        "mem": sample.memory_text,
        "diag": diag
    }
    with open(save_dir + '.json', 'w') as fp:
        json.dump(data, fp, indent = 4)
    print(TP, FP, FN, TN)
    print("reply ratio", Num_trigger_class, Num_trigger_gen, Num_turns)

print(TP, FP, FN, TN)
print("reply ratio", Num_trigger_class, Num_trigger_gen, Num_turns)