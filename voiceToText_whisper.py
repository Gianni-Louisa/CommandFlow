import tkinter as tk
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import threading
import wave
import tempfile
import time as time_module
import os
from pydub import AudioSegment
import datetime
from scipy import signal
from scipy.io import wavfile

try:
    import pyautogui

    pyautogui.FAILSAFE = False
except ImportError:
    print("PyAutoGUI not available - mouse control features disabled")
    pyautogui = None

# Global variables
listening = False
SILENCE_THRESHOLD = 500
SAMPLE_RATE = 48000

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = "voice_recordings"
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

print("Loading Whisper model...")
model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("Model loaded!")


def is_speech(audio_data):
    return np.max(np.abs(audio_data)) > SILENCE_THRESHOLD


def preprocess_audio(audio_data, sample_rate=16000):
    try:
        # Convert to float32 for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Amplify the audio (adjust multiplier as needed)
        audio_float = audio_float * 2

        # Clip to prevent distortion
        audio_float = np.clip(audio_float, -1.0, 1.0)

        # Convert back to int16
        audio_processed = (audio_float * 32767).astype(np.int16)
        return audio_processed

    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        return audio_data


def save_recording_as_mp3(audio_data, recognized_text):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")
        mp3_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.mp3")
        txt_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.txt")

        # Make sure the directory exists
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        # Save WAV file and verify it was created
        wavfile.write(wav_path, SAMPLE_RATE, audio_data)
        if not os.path.exists(wav_path):
            print(f"Failed to create WAV file: {wav_path}")
            return None

        # Convert to MP3 only if WAV exists
        if os.path.exists(wav_path):
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate="320k")

            # Save text file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(recognized_text)

            # Only remove WAV file if MP3 was created successfully
            if os.path.exists(mp3_path):
                os.remove(wav_path)

            return mp3_path
        else:
            print(f"WAV file not found: {wav_path}")
            return None

    except Exception as e:
        print(f"Error saving recording: {e}")
        # If WAV file exists but conversion failed, clean up
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass
        return None


def process_voice_command(command):
    command = command.lower().strip()
    print(f"Processing command: {command}")

    # List of known commands and their variations
    move_mouse_commands = ["move mouse", "move the mouse"]
    exit_commands = ["exit window", "close window"]

    try:
        # Check for move mouse commands
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

        # Check for exit commands
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
        # Preprocess audio for recognition
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

            # Save recording
            mp3_path = save_recording_as_mp3(processed_audio, text)

            # Process command
            if not process_voice_command(text):
                print("Command not recognized")
                status_label.after(0, lambda: status_label.config(text=f"Command not recognized: {text}"))
        else:
            print("No speech detected")
            status_label.after(0, lambda: status_label.config(text="No speech detected"))

        os.unlink(temp_filename)

    except Exception as e:
        print(f"Error in audio processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Processing error: {str(e)}"))


class AudioProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.is_recording = False
        self.silence_count = 0
        self.energy_threshold = SILENCE_THRESHOLD

    def process_audio(self):
        CHUNK = 2048
        MAX_SILENCE_CHUNKS = 8

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
                return

            if listening:
                current_audio = indata.copy().flatten()
                current_energy = np.max(np.abs(current_audio))

                if current_energy > self.energy_threshold:
                    if not self.is_recording:
                        print("Speech detected!")
                    self.is_recording = True
                    self.silence_count = 0
                    self.audio_buffer.append(current_audio)
                elif self.is_recording:
                    self.silence_count += 1
                    self.audio_buffer.append(current_audio)

                    if self.silence_count >= MAX_SILENCE_CHUNKS:
                        if len(self.audio_buffer) > 0:
                            complete_audio = np.concatenate(self.audio_buffer)
                            print("Processing recorded audio...")
                            threading.Thread(target=save_and_process_audio,
                                             args=(complete_audio,)).start()

                        self.audio_buffer = []
                        self.is_recording = False
                        self.silence_count = 0

        try:
            print("Starting audio stream...")
            with sd.InputStream(callback=audio_callback,
                                channels=1,
                                samplerate=SAMPLE_RATE,
                                blocksize=CHUNK,
                                dtype=np.int16,
                                device=None,  # Use default device
                                latency='low') as stream:  # Added low latency
                print("Audio stream started")
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
            print("Recording thread started")

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
