from pyannote.database import ProtocolFile
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pathlib import Path

class MyDataset(SpeakerDiarizationProtocol):

    def trn_iter(self):
        return self._iterate("/Users/ludovicmaitre/Documents/Lilio/mili-o/diarization/finteTunning/synthetic_corpus/train/")

    def dev_iter(self):
        return self._iterate("/Users/ludovicmaitre/Documents/Lilio/mili-o/diarization/finteTunning/synthetic_corpus/test/")

    def _iterate(self, folder):
        audio_folder = Path(folder) / "audio"
        for wav_file in audio_folder.glob("*.wav"):
            uri = wav_file.stem
            yield ProtocolFile(
                uri=uri,
                audio={"uri": wav_file},
                annotation=str(folder / f"rttm/{uri}.rttm")
            )
