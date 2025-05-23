from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
import os
import argparse

# Set up argument parser

os.makedirs("./output", exist_ok=True)

parser = argparse.ArgumentParser(description="Speaker Diarization with Diart")
parser.add_argument(
    "--type",
    type=str,
    choices=["mic", "file"],
    required=True,
)
parser.add_argument(
    "--input",
    type=str,
    help="Input file path for audio",
)
args = parser.parse_args()
filename = "file"
if args.type == "mic":
    mic = MicrophoneAudioSource()
    mic.uri = "microphone"
elif args.type == "file":
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    mic = FileAudioSource(args.input, sample_rate=48000)
    mic.uri = args.input
    filename = args.input.split("/")[-1]

# Initialize the pipeline and inference


pipeline = SpeakerDiarization()
inference = StreamingInference(pipeline, mic, do_plot=True)
inference.attach_observers(RTTMWriter(mic.uri, f"./output/{filename}.rttm"))
prediction = inference()