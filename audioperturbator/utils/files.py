import os
import glob

import pkg_resources

AMBIENT_AUDIO = 'data/245041__kwahmah-02__pub-ambience-1.wav'
EXAMPLE_AUDIO = 'data/410574__yummie__game-background-music-loop-short.mp3'


__all__ = ['example_audio_file', 'ambient_audio_file']


def example_audio_file():
    """Get the path to an audio example file

    Returns:
        str: filename of example audio
    """
    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)


def ambient_audio_file():
    """Get the path to an ambient sound 

    Returns:
        str: filename of ambient audio
    """
    return pkg_resources.resource_filename(__name__, AMBIENT_AUDIO)
