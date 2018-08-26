from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = None # TODO add a pretrained model?
    input_path = os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3")
    output_path = None

@ex.automain
def main(cfg, model_path, input_path, output_path=None):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)