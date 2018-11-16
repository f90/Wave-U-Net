from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "full_44KHz", "full_44KHz-236118") # Load stereo vocal model by default
    input_path = os.path.join("audio_examples", "The Mountaineering Club - Mallory", "mix.mp3") # Which audio file to separate
    output_path = None # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)