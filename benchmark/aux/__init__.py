from benchmark.aux.preprocess import PREPROCESS, PreProcessCommand
from benchmark.aux.predict_chips import PREDICT_CHIPS, PredictChipsCommand

def register_plugin(plugin_registry):
    plugin_registry.register_aux_command(PREPROCESS, PreProcessCommand)
    plugin_registry.register_aux_command(PREDICT_CHIPS, PredictChipsCommand)