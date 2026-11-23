import numpy as np

from autoshorts.transcribe import Word, _is_cuda_error, sentence_ends, speech_density


def test_speech_density_counts_words_per_second():
    words = [Word(0.1, 0.3, "a"), Word(0.3, 0.45, "b"), Word(2.0, 2.4, "c")]
    d = speech_density(words, duration=3.0, hop=0.5)
    assert len(d) == 6
    assert d[0] == 4.0  # two words in the first half second
    assert d[4] == 2.0
    assert d.sum() == 6.0


def test_words_past_the_end_are_ignored():
    assert speech_density([Word(10, 11, "late")], duration=3.0, hop=0.5).sum() == 0


def test_sentence_ends_on_punctuation_and_pauses():
    words = [Word(0, 0.5, "hi."), Word(0.6, 1.0, "so"), Word(1.0, 1.3, "um"), Word(2.5, 3.0, "ok")]
    assert sentence_ends(words) == [0.65, 1.45]


def test_cuda_error_detection():
    assert _is_cuda_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert _is_cuda_error(RuntimeError("Could not load library cudnn_ops64_9.dll"))
    assert not _is_cuda_error(ValueError("bad language code"))


def test_empty_audio_density():
    assert np.all(speech_density([], 1.0, 0.5) == 0)
