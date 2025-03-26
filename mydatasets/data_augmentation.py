import random
import string
from typing import List, Tuple, Callable
import re

SILENT_REPLACE = " @@" #" |SILENCE >"

random.seed(20)

def adapt_to_ASR(dialogue):
    # change to lower case
    dialogue = dialogue.lower()
    punctuation = ['.', '!', '?', ';', ',']
    for p in punctuation:
        dialogue = dialogue.replace(p, "")
    return dialogue


def drop_word(text: str, prob = 0.01) -> str:
    """Randomly delete words"""
    words = text.split()
    aug_text = ""
    for w in words:
        if random.random() > prob or (w.startswith("|")) or (w in ["^^", ">", "@@"]) or ("speaker" in w or  "user" in w or "agent" in w): # drop
            aug_text += (' ' + w)
        # else:
        #     print("drop", w)

    aug_text = aug_text[1:]
    return aug_text

def swap_silence_speaker(text: str, prob = 0.1):
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] == "@@" and ("speaker" in words[i+1] or  "user" in words[i+1]):
            if random.random() < prob:
                # print("!!!!!")
                w = words[i]
                words[i] = words[i+1]
                words[i+1] = w
    
    aug_text = ' '.join(w for w in words)
    return aug_text



def augement_dialogue(dialogue, config):
    dialogue = dialogue.replace(" |SILENCE >", SILENT_REPLACE)
    if "adapt_to_ASR" in config.keys():
        if random.random() < config["adapt_to_ASR"]:
            # print("adapt_to_ASR")
            dialogue = adapt_to_ASR(dialogue)
    if "drop_word" in config.keys():
        if random.random() < config["drop_word"]:
            # print("drop_word")
            dialogue = drop_word(dialogue)
    if "swap_silence_speaker" in config.keys():
        if random.random() < config["swap_silence_speaker"]:
            # print("swap_silence_speaker")
            dialogue = swap_silence_speaker(dialogue)

    dialogue = dialogue.replace(SILENT_REPLACE, " |SILENCE >")
    return dialogue



if __name__ == "__main__":
    aug_config = {
        "adapt_to_ASR": 1,
        "drop_word": 1,
        "swap_silence_speaker": 1
    }

    dialogue = "Speaker1:  So, how are you doing? User: I'm pretty good. |SILENCE > |SILENCE > Doing well. |SILENCE > |SILENCE > |SILENCE > Speaker1: So, please tell me about yourself. |SILENCE > |SILENCE > Okay. |SILENCE > |SILENCE > |SILENCE > User: So, I guess... Have you looked at my resume, or should I? Alright. So, I guess I'm a course 6-7 here at MIT, which is computational biology. So, it's a mix of computer science and biology. And actually, that's where my interests lie, in applying, like, algorithmic kind of software engineering to data sets, dealing with genomics and biology. |SILENCE > |SILENCE > Some of the activities to do outside of school include Camp Kesem, which is a summer camp that we run for completely free for kids whose parents have cancer, as well as Amphibious Achievement, which is a high school tutoring program for inner-city kids in Boston. |SILENCE > |SILENCE > |SILENCE > So, |SILENCE > |SILENCE > my interests kind of lie both in a little bit of the healthcare. I imagined I was going to be a doctor growing up, and then I came to MIT, and I'm like, well, I can do engineering and still apply a lot of these same things and help a lot more people. |SILENCE > |SILENCE > |SILENCE > |SILENCE > Speaker1: So, please tell me about a time you demonstrated leadership. |SILENCE > |SILENCE > Okay. |SILENCE > |SILENCE > User: One of the things that we have to do for Camp Kesem is fundraise all the money to |SILENCE > |SILENCE > run the camp, which is over $50,000. |SILENCE > Agent: Leadership role |SILENCE > And so, one of the things that I individually spearhead every year is called the Camp Kesem SAE Data Auction, where actually my fraternity and I go out and solicit |SILENCE > |SILENCE > |SILENCE > donations in the form of gift cards to raise money for a data auction where we actually sell dates. And then we use this money, obviously, and we donate it to Camp Kesem. So, I spearhead the entire event, and I kind of organize it into committees and groups, and I send the people out and make sure everything |SILENCE > |SILENCE > goes according to plan. |SILENCE > |SILENCE > Speaker1: Tell me about a time when you were working on a team and faced with a challenge. How did you solve that problem? |SILENCE > |SILENCE > |SILENCE > |SILENCE > |SILENCE > |SILENCE > |SILENCE > |SILENCE > |SILENCE > User: I guess the easiest team project I just had was |SILENCE > |SILENCE > last semester, I worked on a 6.005 project, which is software architecture. |SILENCE >."
    aug_dialogue = augement_dialogue(dialogue, aug_config)
    print(dialogue)
    print("-------")
    print(aug_dialogue)
