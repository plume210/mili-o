# 🥧 LlamaPIE
## LlamaPIE🥧: Proactive In-Ear  Conversation  Assistants

<p align="center">
        🤗 <a href="https://huggingface.co/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/xxx">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://huggingface.co">MLX Model</a>
</p>

<p align="center">
    <img src="img/llamapie.png" width="400"/>
<p>

## 💡 Highlight
* In-ear AI system could **proactively** assist users' vocal conversation without explicit invocation
* Proactively determining when to provide assistance and delivering concise, unobtrusive messages, e.g. reminding, guidance, and support.
* Streaming and realtime operation in MLX framework in Apple Silicon

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Training code
- [ ] Realtime on-device speech and text pipelin


# 🔧Quick Start
## Setup environment
```
git clone https://github.com/chentuochao/LlamaPIE.git
conda create -n llamapie python=3.9
pip install -r requirement.txt
```
## Dataset preparation
Training/Val dataset and Test dataset

# 📊 Inference
## Run inference on synthetic dataset
Torch version LlamaPIE🥧 checkpoint available in Huggingface, also we release the MLX version of LlamaPIE🥧 for on-device real-time application. 
```
python infer_dual_model.py
```
## Rubric Score evaluation
We use ChatGPT-4o as our rubric score evaluator,
```
python infer_dual_model.py
```

# 🏋️‍♂️ Training
## Train the small model
```
python train_small.py --save_path /gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_whisper_aware
```

## Train the large model
```
python train_large.py --model llama3_1-8b --data_path /scr/Final_Generation/Pos_Neg/ --save_path /gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_3data
```