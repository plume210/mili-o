from datasets import load_dataset
import datasets
from pathlib import Path
import os
import string
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_DATASETS_CACHE'] = '/scr/data'
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EvalPrediction
from model.CasualTokenClassificationLlama import LlamaForCausalLM_TokenClassifcation
from sklearn.metrics import accuracy_score

model_id = "meta-llama/Llama-3.2-1B-Instruct" 
### loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir = "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
    pad_token="<|eot_id|>",
    token = "xxx"
    )

tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM_TokenClassifcation.from_pretrained(
    model_id,
    device_map='cuda', 
    torch_dtype=torch.bfloat16, 
    num_labels = 2,
    cache_dir = "/gscratch/intelligentsystems/tuochao/Large_Model/llama3_2_1b",
    token = "xxx")

print("loadede!!!!!!!!")

### check the causality of llama3
def check_causality(model, input_ids):
    ### need to sset model dtype to float32
    test_samples = 10
    out1 = model(input_ids = input_ids[:, :test_samples])
    out2 = model(input_ids = input_ids)
    out1 = out1[0][:, :test_samples]
    out2 = out2[0][:, :test_samples]

    print(out1[0])
    print(out2[0])
    print(torch.allclose(out1, out2, rtol=1e-03, atol=1e-03))


### load dataset 
from mydatasets.Active_dataset import New_WhisperAware_dataset, New_WhisperAware_dataset2

positive_base_train = [
    "/gscratch/intelligentsystems/common_datasets/active_agent_fix/Train/claude/",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/perl/Train/claude",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/soda/Train/claude",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/synthetic/Train/claude",
]
positive_base_dev = [
    "/gscratch/intelligentsystems/common_datasets/active_agent_fix/Val/claude/",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/perl/Val/claude",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/soda/Val/claude",
    "/gscratch/intelligentsystems/common_datasets/active_agent/finalData/Inference/synthetic/Val/claude",
]


aug_config = {
    "adapt_to_ASR": 1,
    "drop_word": 0.7,
    "swap_silence_speaker": 0.3
}


parser = argparse.ArgumentParser()
# Experiment Params
parser.add_argument('--save_path', type=str,
                    help='Path to save model')

args = parser.parse_args()

aug_config = None
## whisper awrae aug
save_dir = args.save_path #"/gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_whisper_aware_aug"
negative_base_train = None
negative_base_dev = None
dataset = New_WhisperAware_dataset(tokenizer, input_dirs = positive_base_train, split_set="Train", negative_base=negative_base_train, aug_config = aug_config)                                                
dataset_val = New_WhisperAware_dataset(tokenizer, input_dirs = positive_base_dev, split_set="Val", negative_base=negative_base_dev, aug_config = aug_config)

print(f"Train {len(dataset)} and val {len(dataset_val)}")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["g_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# for sample in loader:
    # inputs = torch.tensor(sample['input_ids']).to(device = 'cuda')
    # print(inputs.shape)
    # output = model(inputs)
    # print(output['logits'].shape)
    # check_causality(model, inputs)
    # outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    # print(outputs)
training_args = TrainingArguments(
    output_dir=save_dir,
    eval_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    report_to= "none",
    num_train_epochs = 10,
)

## define eval metric
def compute_metrics(pred: EvalPrediction):
    predictions, labels, inputs = pred.predictions, pred.label_ids, pred.inputs

    print("predictions", predictions.shape)
    print("labels", labels.shape)
    # print("inputs", inputs.shape)
    predicted_class_indices = np.argmax(predictions, axis=-1)
    flattened_preds = predicted_class_indices.flatten()
    flattened_labels = labels.flatten()
    valid_indices = (flattened_labels != -100)
    flattened_preds = flattened_preds[valid_indices]
    flattened_labels = flattened_labels[valid_indices]
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    ### iterrate with each batch

    diag_correct = 0
    for i in range(0, predictions.shape[0]):
        pred_one = predicted_class_indices[i, :]
        label_one = labels[i, :]
        # input_one = inputs[i, :]
        valid_indices = (label_one != -100)
        flattened_pred = pred_one[valid_indices]
        flattened_label = label_one[valid_indices]
        # print(flattened_label.shape, flattened_label.shape)
        # input_one = input_one[valid_indices]

        accuracy0 = accuracy_score(flattened_label, flattened_pred)
        if accuracy0 == 1:
            diag_correct += 1

    diag_accuracy = diag_correct/predictions.shape[0]
    print(accuracy, diag_accuracy)
    return {'accuracy': accuracy, "diag_accuracy": diag_accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=DataCollatorForTokenClassification(tokenizer=dataset.tokenizer),
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=False)
