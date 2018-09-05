from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = "checkpoints/baseline_stereo/baseline_stereo-186093" # Load stereo vocal model by default
    input_path = os.path.join("audio_examples", "The Mountaineering Club - Mallory", "mix.mp3")
    output_path = None

@ex.automain
def main(cfg, model_path, input_path, output_path=None):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)