import os

from ...utils import import_modules


AUDIO_TOWER_FACTORY = {}

def AudioTowerFactory(audio_tower_name):
    audio_tower_name = audio_tower_name.split(':')[0]
    model = None
    for name in AUDIO_TOWER_FACTORY.keys():
        if name.lower() in audio_tower_name.lower():
            model = AUDIO_TOWER_FACTORY[name]
    assert model, f"{audio_tower_name} is not registered"
    return model


def register_audio_tower(name):
    def register_audio_tower_cls(cls):
        if name in AUDIO_TOWER_FACTORY:
            return AUDIO_TOWER_FACTORY[name]
        AUDIO_TOWER_FACTORY[name] = cls
        return cls
    return register_audio_tower_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.audio_tower")
