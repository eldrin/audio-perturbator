import os
import subprocess
import tempfile

import numpy as np
import librosa
import soundfile as sf
import colorednoise as cn
import pyrubberband as pyrb

from .utils import mix, ambient_audio_file


class BaseTransformer(object):
    """Transformer base

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__()
        self.sample_rate = sample_rate

    def __call__(self, x, *args, **kwargs):
        """Method for transformation

        Args:
            x (numpy.ndarray): input signal (n_samples,)

        Returns:
            numpy.ndarray: output (n_samples,)
        """
        raise NotImplementedError()


class PitchShifter(BaseTransformer):
    """Pitch Shifter

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__(sample_rate)

    def __call__(self, x, shift=1):
        """Shift the pitch of given signal
        
        Args:
            x (numpy.ndarray): input signal (n_samples,)
            shift (float, int): degree of shifting (unit:semitones)

        Returns:
            numpy.ndarray: output (n_samples,)
        """
        y = pyrb.pitch_shift(x, self.sample_rate, shift)
        return y


class TimeStretcher(BaseTransformer):
    """Time stretcher

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__(sample_rate)

    def __call__(self, x, stretch=1):
        """Stretch the time of given signal
        
        Args:
            x (numpy.ndarray): input signal (n_samples,)
            stretch (float, int): degree of stretching (unit:ratio)

        Returns:
            numpy.ndarray: output (n_samples,)
        """
        y = pyrb.time_stretch(x, self.sample_rate, stretch)
        return y


class SoundMixer(BaseTransformer):
    """Class for mixing given input to other sound

    Args:
        other (numpy.ndarray): signal to mix to input
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, other, sample_rate=22050):
        """"""
        super().__init__(sample_rate)
        self.other = other

    def __call__(self, x, snr=30):
        """Stretch the time of given signal
        
        Args:
            x (numpy.ndarray): input signal (n_samples,)
            snr (float, int): mixing ratio (unit:dB) 

        Returns:
            numpy.ndarray: output (n_samples,)
        """
        # check the length between two signals
        if len(self.other) == len(x):
            y = mix(x, self.other, snr)

        elif len(self.other) < len(x):
            # repeat
            rem = len(x) - len(self.other)
            other = np.r_[self.other, self.other[:rem]]
            y = mix(x, self.other, snr)

        elif len(self.other) > len(x):
            # randomly crop
            st = np.random.randint(len(self.other) - len(x))
            y = mix(x, self.other[st:st+len(x)], snr)

        else:
            raise ValueError()

        return y


class PubAmbientMixer(SoundMixer):
    """Class for mixing given input to pub ambient noise

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        # load pub ambient
        noise, _ = librosa.load(ambient_audio_file(), sr=sample_rate)
        super().__init__(noise, sample_rate)


class NoiseMixer(BaseTransformer):
    """Mixing noise to the signal

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__(sample_rate)

    def _generate_noise(self, length):
        """Generate noise to mix to target music

        Args:
            length (int): desired length of noise signal
        """
        raise NotImplementedError()

    def __call__(self, x, snr=30):
        """Mixing given signal with noise
        
        Args:
            x (numpy.ndarray): input signal (n_samples,)
            snr (float, int): mixing ratio (unit:dB) 

        Returns:
            numpy.ndarray: output (n_samples,)
        """
        n = self._generate_noise(len(x))
        y = mix(x, n, snr)
        return y


class PinkNoiseMixer(NoiseMixer):
    """Mixing pink noise to the signal

    Args:
        sample_rate (int, float): sampling rate of the process
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__(sample_rate)

    def _generate_noise(self, length):
        """Generate noise to mix to target music

        Args:
            length (int): desired length of noise signal
        """
        return cn.powerlaw_psd_gaussian(1, length)


class MP3Compressor(BaseTransformer):
    """Compress input audio using FFMPEG
    """
    def __init__(self, sample_rate=22050):
        """"""
        super().__init__(sample_rate)

    def __call__(self, x, kbps=128, return_output=True):
        """Compress the input audio with given kbps (CBR)

        NOTE: referenced bmcfee's pyrubberband
        
        Args:
            x (numpy.ndarray): input signal (n_samples,)
            kbps (int): target kbps
            return_output (bool): flag to return processed PCM

        Returns:
            numpy.ndarray: compressed output (n_samples,)

        .. _pyrubberband:
            https://github.com/bmcfee/pyrubberband
        """
        assert kbps in {
            8, 16, 24, 32, 40, 48, 64, 80, 96,
            112, 128, 160, 192, 224, 256, 320
        }

        # get the input and output tempfile
        fd, infile = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        fd, outfile = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)

        # dump the audio
        librosa.output.write_wav(infile, x, self.sample_rate)

        try:
            # Execute ffmpeg
            arguments = [
                'ffmpeg', '-y',
                '-i', infile,
                '-codec:a', 'libmp3lame',
                '-b:a', '{:d}k'.format(kbps),
                outfile
            ]

            subprocess.check_call(arguments,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)

            # load the audio back
            y, sr = librosa.load(outfile)

        except OSError as exc:
            raise RunTimeError('Failed to execute ffmpeg.'
                               'Please verify that ffmpeg '
                               'is installed.')

        finally:
            # close and remove temp files
            os.unlink(infile)
            os.unlink(outfile)

        return y
