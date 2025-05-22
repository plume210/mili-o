from pyannote.database import registry, FileFinder
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Charger la base AMI à partir du fichier database.yml
registry.load_database("AMI-diarization-setup/pyannote/database.yml")

# Charger le protocole (ex: AMI.SpeakerDiarization.mini)
protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors={"audio": FileFinder()})
# Récupérer un fichier test
test_file = next(protocol.test_iter())  # -> dictionnaire {'uri': ..., 'audio': ..., 'annotation': ...}

# Charger le modèle préentraîné de segmentation
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

# Créer l'objet d'inférence
inference = Inference(model, step=2.5)  # step peut être ajusté selon ton besoin


def test(model, protocol, subset="test"):
    from pyannote.audio.utils.signal import binarize
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.pipelines.utils import get_devices

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device)

    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)

    return abs(metric)

der_pretrained = test(model=model, protocol=protocol, subset="test")
print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
# Affichage des résultats
