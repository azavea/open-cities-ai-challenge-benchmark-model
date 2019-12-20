from benchmark.aux.preprocess import PREPROCESS, PreProcessCommand
from benchmark.aux.postprocess import POSTPROCESS, PostProcessCommand

def register_plugin(plugin_registry):
    plugin_registry.register_aux_command(PREPROCESS, PreProcessCommand)
    plugin_registry.register_aux_command(POSTPROCESS, PostProcessCommand)