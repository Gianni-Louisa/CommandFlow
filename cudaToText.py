"""

Code Artifact: cudaToText.py
Description: Program to listen for audio input and handle commands

@author: Gianni Louisa, Connor Bennudriti, Ethan Dirkes, Christoper Gronewold, Tommy Lam
@created: 2/14/2025
@revised: 3/2/2025

Revision History:
- 2/14/2025: Initial creation of script
- 3/2/2025 (Ethan Dirkes): Added image to record button and label to display detected speech
- 3/2/2025: Commented code

Preconditions:
- OpenAI's whisper library must be installed
- Program must be run on Nvidia GPU with CUDA in order for the CUDA functionality to work (but is not needed for code execution)

Postconditions:
- Displays audio on a GUI
- Performs actions that are commanded by the user in the audio

"""

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
    # Try to import pyautogui for mouse control functionality
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable the failsafe feature that stops mouse movement when cursor hits screen corner
except ImportError:
    # Handle the case where pyautogui is not installed
    print("PyAutoGUI not available - mouse control features disabled")
    pyautogui = None

# Global configuration and constants
SILENCE_THRESHOLD = 500 # Energy threshold to determine when speech is occurring
SAMPLE_RATE = 48000 # Audio sampling rate in Hz

# Threading event to control when the app is actively listening
listening_event = threading.Event()

# Create a thread pool to handle background processing tasks
executor = ThreadPoolExecutor(max_workers=4)

# Determine whether to use GPU or CPU for processing
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise use CPU **ONLY NVIDIA GPU**

# Initialize the speech recognition model, using English small model with int8 quantization
print("Loading Whisper model...")
model = WhisperModel("small.en", device=device, compute_type="int8") 
print("Model loaded!")


def preprocess_audio(audio_data, sample_rate=16000):
    """
    proprocess_audio: Function to process audio so it is more clear
    """
    try:
        # Convert from int16 to float32 and normalize to [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_float *= 2 # Amplify the signal
        np.clip(audio_float, -1.0, 1.0, out=audio_float) # Clip to avoid distortion
        # Convert back to int16 format
        audio_processed = (audio_float * 32767).astype(np.int16)
        return audio_processed
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        return audio_data # Return original data if processing fails


def process_voice_command(command):
    """
    process_voice_command(): Function to perform the command that was heard by the audio listener
    """
    command = command.lower().strip() # Normalize command to lowercase and remove whitespace
    print(f"Processing command: {command}")
    # Define command categories for easier matching
    move_mouse_commands = ["move mouse", "move the mouse"]
    exit_commands = ["exit window", "close window"]

    try:
        # Hadle mouse movement commands
        if any(cmd in command for cmd in move_mouse_commands):
            if "top right" in command:
                # Move mouse to top-right corner of screen
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))
                screen_width, _ = pyautogui.size() # Get screen dimensions
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5) # Move with animation over 0.5 seconds
                return True
            else:
                # Move mouse to default position if no specific location mentioned
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position"))
                icon_x, icon_y = 200, 200 # Default position coordinates
                pyautogui.moveTo(icon_x, icon_y, duration=0.5) # Move with animation over 0.5 seconds
                return True

        # Handle window closing commands
        if any(cmd in command for cmd in exit_commands):
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))
            pyautogui.hotkey("alt", "f4") # Simulate Alt+F4 keyboard shortcut
            return True

        return False

    # If there is an error, display info about it
    except Exception as e:
        print(f"Error in command processing: {e}")
        status_label.after(0, lambda: status_label.config(text=f"Command error: {str(e)}"))
        return False


def save_and_process_audio(audio_data):
    """
    save_and_process_audio(): Function to save the audio to a wav file and then process it
    """
    try:
        # Preprocess the audio for better recognition
        processed_audio = preprocess_audio(audio_data)
        # Create a temporary WAV file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_filename = temp_audio_file.name
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio)

        # Transcribe the audio using the Whisper model
        print("Processing audio with Whisper...")
        segments, _ = model.transcribe(
            temp_filename,
            beam_size=5, # Beam search size for more accurate transcription
            language="en", # Force English language
            condition_on_previous_text=True, # Use context from previous segments
            no_speech_threshold=0.3 # Threshold for filtering out non-speech
        )
        # Combine all segments into a single text
        text = " ".join([segment.text for segment in segments])
        # If text was recognized
        if text.strip():
            print(f"Recognized text: {text}")
            # Update UI with recognized text (thread-safe using after)
            recorded_label.after(0, lambda: recorded_label.config(text=f"You said: \"{text.lstrip()}\""))
            # Process any commands in the recognized text
            process_voice_command(text)
        # If no text was recognized
        else:
            print("No speech detected")
            recorded_label.after(0, lambda: recorded_label.config(text="No speech detected")) # display an error message

        # Clean up temporary file
        os.unlink(temp_filename)

    # Handle error by displaying info
    except Exception as e:
        print(f"Error in audio processing: {e}")
        recorded_label.after(0, lambda: recorded_label.config(text=f"Processing error: {str(e)}"))


class AudioProcessor:
    """
    AudioProcessor: A class to process the user's command audio
    """
    def __init__(self):
        self.audio_queue = queue.Queue() # Queue to store incoming audio chunks
        self.audio_buffer = [] # Buffer to accumulate audio during speech
        self.recording_active = False # Flag to track if speech is currently being recorded
        self.silence_count = 0 # Counter to track consecutive silent chunks
        self.energy_threshold = SILENCE_THRESHOLD # Threshold to distinguish speech from silence

    def audio_callback(self, indata, frames, time_info, status):
        """
        audio_callback: A callback method for the sounddevice InputStream
        """
        if status:
            print(f"Audio callback status: {status}") # Log any issues with audio input
        # Only process audio if listening is enabled
        if listening_event.is_set():
            self.audio_queue.put(indata.copy().flatten()) # Flatten to 1D array and copy to queue

    def process_audio(self):
        """
        process_audio: Main class method to process the audio
        """
        CHUNK = 8192 # Size of each audio chunk in samples
        MAX_SILENCE_CHUNKS = 8 # Number of silent chunks to wait before processing (determines pause length)# Number of silent chunks to wait before processing (determines pause length)

        try:
            print("Starting audio stream...")
            # Create and start the audio input stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK,
                dtype=np.int16,
                latency='low' # Low latency for responsive detection
            ) as stream: 
                print("Audio stream started")

                # Main processing loop
                while listening_event.is_set():
                    # Get the next audio chunk from the queue (with timeout to prevent blocking)
                    try:
                        current_audio = self.audio_queue.get(timeout=0.15)
                    except queue.Empty: # If queue is empty, try again
                        continue

                    # Calculate audio energy (maximum absolute amplitude)
                    energy = np.max(np.abs(current_audio))

                    # If energy exceeds threshold, we detected speech
                    if energy > self.energy_threshold:
                        if not self.recording_active:
                            print("Speech detected!")
                            self.recording_active = True # Start recording session
                            self.silence_count = 0 # Reset silence counter
                        self.audio_buffer.append(current_audio) # Add chunk to buffer

                    # If we're already recording but current chunk is silent
                    elif self.recording_active:
                        self.audio_buffer.append(current_audio) # Add silent chunk to buffer
                        self.silence_count += 1 # Increment silence counter
                        
                        # If enough consecutive silent chunks, process the complete utterance
                        if self.silence_count >= MAX_SILENCE_CHUNKS:
                            # Combine all buffered audio chunks
                            complete_audio = np.concatenate(self.audio_buffer)
                            print("Processing recorded audio...")

                            # Submit the processing task to the thread pool
                            executor.submit(save_and_process_audio, complete_audio)
                            
                            # Reset recording state
                            self.audio_buffer = []
                            self.recording_active = False
                            self.silence_count = 0

        except Exception as e:
            print(f"Error in audio recording: {e}")
            # Update UI with error message (thread-safe)
            recorded_label.after(0, lambda: recorded_label.config(text=f"Recording error: {str(e)}"))


def toggle_record():
    """
    toggle_record(): Function to turn audio recording off/on
    """
    if not listening_event.is_set():
        try:
            listening_event.set() # Start listening
            
            # Update UI to show listening state
            status_label.config(text="Speak your command") 
            recorded_label.config(text="Listening...")
            record_button.config(text="Stop Recording")

            # Create and start the audio processing in a background thread
            audio_processor = AudioProcessor()

            # Start a new thread for audio processing
            recording_thread = threading.Thread(target=audio_processor.process_audio, daemon=True) 
            recording_thread.start()
            print("Recording thread started")

        # Handle errors  by logging to GUI
        except Exception as e:
            print(f"Error starting recording: {e}")
            status_label.config(text=f"Error: {str(e)}")
            record_button.config(text="Record")
            recorded_label.config(text="")
            listening_event.clear()
    else:
        # Stop listening
        listening_event.clear() # Clear the event to stop audio processing
        # Update UI to show stopped state
        status_label.config(text="Press the microphone button and speak")
        record_button.config(text="Record")
        recorded_label.config(text="Stopped listening")
        print("Stopped listening")


# Set up the main Tkinter window
root = tk.Tk()
root.title("CommandFlow") # Set application title
root.geometry("480x270") # Set window size

# Create label for current recording status
status_label = tk.Label(root, text="Press the microphone button and speak", wraplength=300, foreground="black")
status_label.pack(pady=20) # Add padding for spacing

# Create button to begin/end recording
mic_image = tk.PhotoImage(file="mic-icon.png").subsample(2,2)   # Get mic image, scale down by 2x
record_button = tk.Button(root, text="Record", image=mic_image, border=0, command=toggle_record)
record_button.pack()

# Create label to write back detected voice input
recorded_label = tk.Label(root, text="", wraplength=300, foreground="#666666")
recorded_label.pack(pady=20)

# Start the main application loop
root.mainloop()
