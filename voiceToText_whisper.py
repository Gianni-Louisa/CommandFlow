import tkinter as tk
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import threading
import tempfile
import os
from scipy.io import wavfile
import queue
from concurrent.futures import ThreadPoolExecutor
import torch

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    print("PyAutoGUI not available - mouse control features disabled")
    pyautogui = None

# Global configuration and constants
SILENCE_THRESHOLD = 500
SAMPLE_RATE = 48000

listening_event = threading.Event()

executor = ThreadPoolExecutor(max_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper model...")
model = WhisperModel("small.en", device=device, compute_type="int8")
print("Model loaded!")


def preprocess_audio(audio_data, sample_rate=16000):
    try:
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_float *= 2
        np.clip(audio_float, -1.0, 1.0, out=audio_float)
        audio_processed = (audio_float * 32767).astype(np.int16)
        return audio_processed
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        return audio_data


def process_voice_command(command):
    command = command.lower().strip()
    print(f"Processing command: {command}")
    move_mouse_commands = ["move mouse", "move the mouse"]
    exit_commands = ["exit window", "close window"]

    try:
        if any(cmd in command for cmd in move_mouse_commands):
            if "top right" in command:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))
                screen_width, _ = pyautogui.size()
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5)
                return True
            else:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position"))
                icon_x, icon_y = 200, 200
                pyautogui.moveTo(icon_x, icon_y, duration=0.5)
                return True

        if any(cmd in command for cmd in exit_commands):
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))
            pyautogui.hotkey("alt", "f4")
            return True

        return False

    except Exception as e:
        print(f"Error in command processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Command error: {str(e)}"))
        return False


def save_and_process_audio(audio_data):
    try:
        processed_audio = preprocess_audio(audio_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_filename = temp_audio_file.name
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio)

        print("Processing audio with Whisper...")
        segments, _ = model.transcribe(
            temp_filename,
            beam_size=5,
            language="en",
            condition_on_previous_text=True,
            no_speech_threshold=0.3
        )
        text = " ".join([segment.text for segment in segments])
        if text.strip():
            print(f"Recognized text: {text}")
            status_label.after(0, lambda: status_label.config(text=f"You said: {text}"))
            process_voice_command(text)
        else:
            print("No speech detected")
            status_label.after(0, lambda: status_label.config(text="No speech detected"))

        os.unlink(temp_filename)

    except Exception as e:
        print(f"Error in audio processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Processing error: {str(e)}"))


class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.recording_active = False
        self.silence_count = 0
        self.energy_threshold = SILENCE_THRESHOLD

    def audio_callback(
        self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        if listening_event.is_set():
            self.audio_queue.put(indata.copy().flatten())

    def process_audio(self):
        CHUNK = 4096
        MAX_SILENCE_CHUNKS = 8

        try:
            print("Starting audio stream...")
            with sd.InputStream(callback=self.audio_callback,
                                channels=1,
                                samplerate=SAMPLE_RATE,
                                blocksize=CHUNK,
                                dtype=np.int16,
                                latency='low') as stream:
                print("Audio stream started")
                while listening_event.is_set():
                    try:
                        current_audio = self.audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    energy = np.max(np.abs(current_audio))
                    if energy > self.energy_threshold:
                        if not self.recording_active:
                            print("Speech detected!")
                            self.recording_active = True
                            self.silence_count = 0
                        self.audio_buffer.append(current_audio)
                    elif self.recording_active:
                        self.audio_buffer.append(current_audio)
                        self.silence_count += 1
                        if self.silence_count >= MAX_SILENCE_CHUNKS:
                            complete_audio = np.concatenate(self.audio_buffer)
                            print("Processing recorded audio...")
                            # Submit the processing task to the thread pool
                            executor.submit(save_and_process_audio, complete_audio)
                            self.audio_buffer = []
                            self.recording_active = False
                            self.silence_count = 0

        except Exception as e:
            print(f"Error in audio recording: {e}")
            status_label.after(0, lambda: status_label.config(text=f"Recording error: {str(e)}"))


def toggle_record():
    if not listening_event.is_set():
        try:
            listening_event.set()
            status_label.config(text="Listening... (Speak now)")
            record_button.config(text="Stop Recording")
            audio_processor = AudioProcessor()
            recording_thread = threading.Thread(target=audio_processor.process_audio, daemon=True)
            recording_thread.start()
            print("Recording thread started")
        except Exception as e:
            print(f"Error starting recording: {e}")
            status_label.config(text=f"Error: {str(e)}")
            record_button.config(text="Record")
            listening_event.clear()
    else:
        listening_event.clear()
        record_button.config(text="Record")
        status_label.config(text="Stopped listening")
        print("Stopped listening")


# Set up the main Tkinter window
root = tk.Tk()
root.title("Voice to Text Demo")
root.geometry("400x200")

status_label = tk.Label(root, text="Press the button and speak.", wraplength=300)
status_label.pack(pady=20)

record_button = tk.Button(root, text="Record", command=toggle_record)
record_button.pack()

root.mainloop()
