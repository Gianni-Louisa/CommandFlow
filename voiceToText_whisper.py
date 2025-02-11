import tkinter as tk
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import threading
import wave
import tempfile
import time as time_module

try:
    import pyautogui

    pyautogui.FAILSAFE = False
except ImportError:
    print("PyAutoGUI not available - mouse control features disabled")
    pyautogui = None

# Global variables
listening = False
audio_buffer = []
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0

print("Loading Whisper model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Model loaded!")


def is_silence(audio_data):
    return np.max(np.abs(audio_data)) < SILENCE_THRESHOLD


def process_voice_command(command):
    command = command.lower().strip()
    print(f"Processing command: {command}")
    try:
        if pyautogui and ("exit window" in command or "close window" in command or "quit" in command):
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))
            pyautogui.hotkey("alt", "f4")
        elif pyautogui and "move mouse" in command:
            if "top right" in command:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))
                screen_width, _ = pyautogui.size()
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5)
            else:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position"))
                icon_x, icon_y = 200, 200
                pyautogui.moveTo(icon_x, icon_y, duration=0.5)
    except Exception as e:
        print(f"Error in command processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Command error: {str(e)}"))


def save_and_process_audio(audio_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_filename = temp_audio_file.name

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())

        print("Processing audio with Whisper...")
        segments, _ = model.transcribe(temp_filename)
        text = " ".join([segment.text for segment in segments])

        if text.strip():
            print(f"Recognized text: {text}")
            status_label.after(0, lambda: status_label.config(text=f"You said: {text}"))
            process_voice_command(text)

    except Exception as e:
        print(f"Error in audio processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Processing error: {str(e)}"))


class AudioProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.last_sound_time = time_module.time()
        self.recording_speech = False

    def process_audio(self):
        CHUNK = 1600
        RATE = 16000

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
                return

            if listening:
                current_audio = indata.copy()

                if not is_silence(current_audio):
                    self.last_sound_time = time_module.time()
                    self.recording_speech = True
                    self.audio_buffer.append(current_audio)
                elif self.recording_speech:
                    self.audio_buffer.append(current_audio)

                    if time_module.time() - self.last_sound_time > SILENCE_DURATION:
                        if len(self.audio_buffer) > 0:
                            complete_audio = np.concatenate(self.audio_buffer)
                            threading.Thread(target=save_and_process_audio,
                                             args=(complete_audio,)).start()

                        self.audio_buffer = []
                        self.recording_speech = False

        try:
            with sd.InputStream(callback=audio_callback,
                                channels=1,
                                samplerate=RATE,
                                blocksize=CHUNK,
                                dtype=np.int16):
                while listening:
                    sd.sleep(100)

        except Exception as e:
            print(f"Error in audio recording: {e}")
            status_label.after(0, lambda: status_label.config(text=f"Recording error: {str(e)}"))


def toggle_record():
    global listening

    if not listening:
        try:
            listening = True
            status_label.config(text="Listening... (Speak now)")
            record_button.config(text="Stop Recording")

            audio_processor = AudioProcessor()
            recording_thread = threading.Thread(target=audio_processor.process_audio)
            recording_thread.daemon = True
            recording_thread.start()

        except Exception as e:
            print(f"Error starting recording: {e}")
            status_label.config(text=f"Error: {str(e)}")
            record_button.config(text="Record")
            listening = False

    else:
        listening = False
        record_button.config(text="Record")
        status_label.config(text="Stopped listening")
        print("Stopped listening")


# Set up the main Tkinter window
root = tk.Tk()
root.title("Voice to Text Demo")
root.geometry("400x200")

# Add a label to display instructions or results
status_label = tk.Label(root, text="Press the button and speak.", wraplength=300)
status_label.pack(pady=20)

# Add a button to start/stop recording
record_button = tk.Button(root, text="Record", command=toggle_record)
record_button.pack()

root.mainloop()
