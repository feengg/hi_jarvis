import pyaudio
from openwakeword.model import Model
import numpy as np
from config import OPEN_WAKE_WORD_MODEL_NAME_LIST, OPEN_WAKE_WORD_MODEL_THREAD_HOLD
from threading import Thread

class WakeWordService():
    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.open_wake_word = Model(
            wakeword_models=[""],
            device = "cpu"
        )
        self.running = False

    def __do_start(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            frames_per_buffer=1280
        )
        self.open_wake_word.reset()
        self.stream.start_stream()
        while self.running and self.stream.is_active():        
            stream_data = self.stream.read(1280)
            sample_data = np.frombuffer(stream_data, dtype=np.int8)
            prediction = self.open_wake_word.predict(sample_data)
            hit = False
            for key_word in OPEN_WAKE_WORD_MODEL_NAME_LIST:
                if prediction[key_word] > OPEN_WAKE_WORD_MODEL_THREAD_HOLD:
                    hit = True
                    break
            if hit:
                break
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        return hit

    def start(self):
        self.running = True
        task = Thread(self.__do_start)
        task.start()

    def stop(self):
        self.running = False

