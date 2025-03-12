# LlamaPIE
Github repo for paper: LlamaPIE: Proactive In-Ear  Conversation  Assistants


# Inference

# Training
## train the small model
```
python train_small.py --save_path /gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/classifier_4data_whisper_aware
```

## train the large model
```
python train_large.py --model llama3_1-8b --data_path /scr/Final_Generation/Pos_Neg/ --save_path /gscratch/intelligentsystems/tuochao/Proactive_Agent/experiment/generator_3data
```