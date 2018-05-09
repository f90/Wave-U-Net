import librosa
import os, re, subprocess
from soundfile import SoundFile
import scikits.audiolab

class AudioReadError(EnvironmentError):
    pass

_find_sampling_rate = re.compile('.* ([0-9:]+) Hz,', re.MULTILINE )
_find_channels = re.compile(".*Hz,( .*?),", re.MULTILINE)
_find_duration = re.compile('.*Duration: ([0-9:]+)', re.MULTILINE )

def timestamp_to_seconds( ms ):
    "Convert a hours:minutes:seconds string representation to the appropriate time in seconds."
    a = ms.split(':')
    assert 3 == len( a )
    return float(a[0]) * 3600 + float(a[1]) * 60 + float(a[2])

def seconds_to_min_sec( secs ):
    "Return a minutes:seconds string representation of the given number of seconds."
    mins = int(secs) / 60
    secs = int(secs - (mins * 60))
    return "%d:%02d" % (mins, secs)

def get_metadata_by_loading(audio_path):
    print("Reading metadata for file " + audio_path + " by loading file completely")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if sr == None: # Error reading file
        raise AudioReadError("Could not load file" + audio_path)
    if len(audio.shape) == 1:
        return sr, 1, float(audio.shape[0]) / float(sr)
    else:
        return sr, audio.shape[0], float(audio.shape[1]) / float(sr)

def get_mp3_metadata(audio_path):
    "Determine length of tracks listed in the given input files (e.g. playlists)."
    ffmpeg = subprocess.check_output(
      'ffmpeg -i "%s"; exit 0' % audio_path,
        shell=True,
      stderr = subprocess.STDOUT )

    # Get sampling rate
    match = _find_sampling_rate.search( ffmpeg )
    if not match:
        return get_metadata_by_loading(audio_path)
    sampling_rate = int(match.group( 1 ))

    # Get channels
    match = _find_channels.search( ffmpeg )
    if not match:
        return get_metadata_by_loading(audio_path)
    channels = match.group( 1 )
    channels = (2 if channels.__contains__("stereo") else 1)

    # Get duration
    match = _find_duration.search( ffmpeg )
    if not match:
        return get_metadata_by_loading(audio_path)
    duration = match.group( 1 )
    duration = timestamp_to_seconds(duration)

    return sampling_rate, channels, duration

def get_audio_metadata(audioPath, sphereType=False):
    '''
    Returns sampling rate, number of channels and duration of an audio file
    :param audioPath: 
    :param sphereType: 
    :return: 
    '''
    ext = os.path.splitext(audioPath)[1][1:].lower()
    if ext=="aiff" or sphereType:  # SPHERE headers for the TIMIT dataset
        audio = scikits.audiolab.Sndfile(audioPath)
        sr = audio.samplerate
        channels = audio.channels
        duration = float(audio.nframes) / float(audio.samplerate)
    elif ext=="mp3": # Use ffmpeg/ffprobe
        sr, channels, duration = get_mp3_metadata(audioPath)
    else:
        snd_file = SoundFile(audioPath, mode='r')
        inf = snd_file._info
        sr = inf.samplerate
        channels = inf.channels
        duration = float(inf.frames) / float(inf.samplerate)
    return int(sr), int(channels), float(duration)
