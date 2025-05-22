import os
from pydub import AudioSegment
import soundfile as sf
from tqdm import tqdm
import torch
from kokoro import KPipeline
import random


#config
output_root = "synthetic_corpus"
os.makedirs(output_root, exist_ok=True)
splits = ["train", "test"]
split_ratio = 0.8  # 80% train, 20% test
sample_rate = 22050
pipeline_FR = KPipeline(lang_code='f')
pipeline_ANG = KPipeline(lang_code='a')
# 2. Créer un répertoire de sortie
os.makedirs("synthetic_corpus/train/audio", exist_ok=True)
os.makedirs("synthetic_corpus/train/rttm", exist_ok=True)

# Définir les locuteurs et leurs phrases
speakers = {
    "spk1": [
        ("fr", "Bonjour, comment vas-tu aujourd'hui ?"),
        ("fr", "Tu as vu le mail que j'ai envoyé ?"),
        ("fr", "Peux-tu m'aider avec ce problème ?"),
        ("fr", "Hmm... ouais, peut-être."),
        ("fr", "Oui, je comprends."),
        ("fr", "Euh... attends une seconde."),
        ("fr", "Ah d'accord, je vois."),
        ("fr", "Oui. Non. Peut-être."),
        ("fr", "Alors. Donc. Bon."),
        ("fr", "C'est une bonne question, je vais y réfléchir."),
        ("fr", "Il faut qu'on se parle rapidement."),
        ("fr", "Tu veux qu'on fasse ça maintenant ou plus tard ?"),
        ("fr", "Je suis d'accord avec toi sur ce point."),
        ("fr", "Ce n’est pas aussi simple que ça en a l’air."),
        ("fr", "Tu peux répéter, s’il te plaît ?"),
        ("fr", "J’ai besoin de plus d’infos pour décider."),
        ("fr", "On peut en discuter après la réunion."),
        ("fr", "Merci pour ton retour."),
        ("fr", "Je pense que tu as raison."),
        ("fr", "Ça marche, je m’en occupe."),
        ("fr", "Alors voilà, ce matin j’ai commencé par relire le compte rendu de la réunion d’hier, puis j’ai tenté de clarifier les points de blocage pour l’équipe. J’ai rédigé une synthèse que je t’ai envoyée par mail. Si tu peux y jeter un œil, ce serait top."),
        ("fr", "Je voulais aussi te dire que j’ai réfléchi à notre organisation actuelle, et je pense qu’on pourrait être plus efficaces si on ajustait un peu la manière dont on priorise nos tâches. Il faudrait peut-être qu’on en reparle tous ensemble d’ici la fin de la semaine.")
    ],
    "spk2": [
        ("en", "Hey, what's the status of the project?"),
        ("en", "I’ll be joining the call at 3pm."),
        ("en", "Did you finish the report?"),
        ("en", "Uh-huh."),
        ("en", "Sure."),
        ("en", "Wait... what?"),
        ("en", "Yep. No. Maybe."),
        ("en", "Right. Got it."),
        ("en", "I'm not sure I follow."),
        ("en", "Let's sync up after lunch."),
        ("en", "That's interesting."),
        ("en", "Actually, that's a good point."),
        ("en", "Let me double-check."),
        ("en", "Could you clarify that?"),
        ("en", "Sounds good to me."),
        ("en", "It's a bit more complicated."),
        ("en", "I'll keep you posted."),
        ("en", "Thanks for the update."),
        ("en", "We need to be more precise."),
        ("en", "Let's not rush this."),
        ("en", "So basically, I went over the analytics again this morning and noticed a consistent drop in user engagement after the onboarding sequence. It could be due to poor UX, but it might also be linked to the lack of follow-up notifications. I’ve listed a few hypotheses we could test in the next sprint."),
        ("en", "To be honest, I don’t think launching the new feature next week is a good idea. The team’s already overwhelmed, and we haven’t finished the regression testing. Let’s take a step back and reassess what really needs to be done before going live.")
    ],
    "spk3": [
        ("fr", "Je ne suis pas sûr de la réponse."),
        ("fr", "On pourrait en discuter ensemble."),
        ("fr", "As-tu des nouvelles de l'équipe ?"),
        ("fr", "Mmh-mmh."),
        ("fr", "Ok."),
        ("fr", "Tu crois ?"),
        ("fr", "Oui oui."),
        ("fr", "C'est noté."),
        ("fr", "Je vais vérifier ça."),
        ("fr", "Ce n'est pas très clair."),
        ("fr", "On doit revoir nos priorités."),
        ("fr", "Tu as une meilleure idée ?"),
        ("fr", "Ça dépend de plusieurs choses."),
        ("fr", "On peut en parler demain."),
        ("fr", "Je ne suis pas dispo tout de suite."),
        ("fr", "Tu peux me rappeler ?"),
        ("fr", "Merci, c’est gentil."),
        ("fr", "Je dois y réfléchir encore un peu."),
        ("fr", "On en reparle plus tard ?"),
        ("fr", "Fais-moi signe quand tu es prêt."),
        ("fr", "J’ai pris un moment pour relire les consignes, et franchement, il y a pas mal de zones d’ombre. Je pense qu’on gagnerait à poser quelques questions directement au client, histoire d’éviter les malentendus. Je vais les lister et les envoyer cet après-midi."),
        ("fr", "Depuis quelques jours, j’essaie de trouver une autre façon de structurer nos livrables. L’idée serait de mieux séparer les blocs fonctionnels, pour que chaque membre de l’équipe sache exactement quoi produire et quand. Ça rendrait notre travail beaucoup plus fluide à mon avis.")
    ],
    "spk4": [
        ("en", "Let's try a different approach next time."),
        ("en", "I'll check the numbers again later."),
        ("en", "Can you send me the updated report?"),
        ("en", "Huh?"),
        ("en", "Okay."),
        ("en", "Makes sense."),
        ("en", "All right."),
        ("en", "That’s interesting."),
        ("en", "Here’s what I think."),
        ("en", "We should take a step back."),
        ("en", "Let’s regroup tomorrow."),
        ("en", "I’ll write a summary."),
        ("en", "Could you please elaborate?"),
        ("en", "Not exactly what I meant."),
        ("en", "Let’s not jump to conclusions."),
        ("en", "This needs more thinking."),
        ("en", "I’ll take care of that."),
        ("en", "Appreciate the input."),
        ("en", "Let’s wrap this up."),
        ("en", "Keep me in the loop."),
        ("en", "The client feedback was pretty clear on what they expect, and I think we need to revise both the timeline and the resource allocation. I've put together a quick plan to adjust, which I’ll walk you through during the meeting. We really can’t afford another delay."),
        ("en", "After the last deployment, I noticed some discrepancies between staging and production. I spent the entire morning comparing logs, and I think the issue lies in how we handle config values per environment. We’ll need to patch that before the next release.")
    ]
}



# Mélange les phrases
all_samples = []
for spk, entries in speakers.items():
    for lang, text in entries:
        all_samples.append((spk, lang, text))

random.shuffle(all_samples)

# Split train / test
split_index = int(len(all_samples) * split_ratio)
print(f"Nombre total d'échantillons : {len(all_samples)}")
print(split_index)
data_splits = {
    "train": all_samples[:split_index],
    "test": all_samples[split_index:]
}

voices = {
    "spk1" : "af_alloy",
    "spk2": "af_sky",
    "spk3":"af_heart",
    "spk4":"ef_dora",
}

for split in splits:
    audio_dir = os.path.join(output_root, split, "audio")
    rttm_dir = os.path.join(output_root, split, "rttm")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    iteration =0
    rttm_lines = []
    combined = AudioSegment.silent(duration=1000)
    start_time = 1.0
    print(f"🔊 Génération des voix pour {split}...")
    print(f"Nombre d'échantillons : {len(data_splits[split])}")
    for i, (spk) in enumerate(tqdm(data_splits[split])):
        print(f"🔊 Génération de {spk}... et text {spk[2]}")
        if spk[1] == "fr":
            generator = pipeline_FR(spk[2], voice=voices[spk[0]])
        else:
            generator = pipeline_ANG(spk[2], voice=voices[spk[0]])
        filename = f"temp_{spk[1]}.wav"
        for i, (gs, ps, audio) in enumerate(generator):
            duration = len(audio) / 24000
            sf.write(filename, audio, 24000)
        audio = AudioSegment.from_wav(filename)
        # RTTM compatible info
        rttm_lines.append(f"SPEAKER dialogue1 1 {start_time:.2f} {duration:.2f} <NA> <NA> {spk[0]} <NA> <NA>")
        combined += audio
        start_time += duration 
        # si la durée totale dépasse 120 secondes, sauvegarder le fichier
        if len(combined) > 120000:
            # 4. Sauvegarder le fichier final
            combined.export(f"synthetic_corpus/{split}/audio/dialogue{iteration}.wav", format="wav", )
            iteration += 1
            with open(f"synthetic_corpus/{split}/rttm/dialogue{iteration}.rttm", "w") as f:
                for line in rttm_lines:
                    f.write(line + "\n")
            # 5. Supprimer le fichier temporaire
            # os.remove(filename)
    print(f"✅ Fichier {split} généré avec succès !")

print("✅ Données synthétiques générées pour l'entraînement et le test.")

segments = []
combined = AudioSegment.silent(duration=1000)

start_time = 1.0
print("🔊 Génération des voix...")


print("✅ Données synthétiques prêtes !")
