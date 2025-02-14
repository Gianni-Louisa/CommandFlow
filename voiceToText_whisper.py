# Filename: voiceToText.py
# Description: Voice to Text Demo using SpeechRecognition, Tkinter, and PyAutoGUI.
#              This script provides a simple GUI that allows the user to record voice,
#              convert it to text using the Google Speech Recognition API, and update the
#              interface with the recognized text. In addition, it processes specific voice
#              commands to control the computer (e.g., exit a window, move the mouse to a specific position).
#
# Programmer: Chris Gronwold, Gianni Louisa
# Date Created: 2025-02-10
# Last Revised: 2025-02-10
#
# Revision History:
#
#
# Preconditions:
#   - Python 3.x installed.
#   - Required packages installed: tkinter, SpeechRecognition, PyAudio, pyautogui.
#   - A functioning microphone connected and set as the default input device.
#
# Acceptable Input Values/Types:
#   - Speech input from a valid microphone.
#
# Unacceptable Input Values/Types:
#   - Silence or non-speech ambient noise may lead to recognition errors.
#
# Postconditions:
#   - The GUI is updated with the recognized speech text.
#   - Recognized voice commands trigger computer control actions.
#
# Return Values:
#   - This script does not return any values; it updates the GUI interface and performs actions based on recognized speech.
#
# Error/Exception Conditions:
#   - sr.UnknownValueError: Thrown when the speech is unintelligible.
#   - sr.RequestError: Thrown when there is an issue with the Google Speech Recognition API request.
#   - OSError: Occurs if no default microphone is available.
#   - sr.WaitTimeoutError: Occurs if no speech is detected within the specified timeout period.
#
# Side Effects:
#   - Opens a GUI window.
#   - Utilizes system audio resources (microphone).
#   - Controls the computer (e.g., exits windows, moves the mouse).
#
# Invariants:
#   - The variable "listening" accurately reflects whether background listening is active.
#
# Known Faults:
#   - Real-time transcription updates occur only after a phrase has been completed, rather than continuously.

import tkinter as tk # Import tkinter for GUI
from faster_whisper import WhisperModel # Import WhisperModel for speech recognition
import sounddevice as sd # Import sounddevice for audio input
import numpy as np # Import numpy for numerical operations
import threading # Import threading for multithreading
import wave # Import wave for audio file operations
import tempfile # Import tempfile for temporary file operations
import time as time_module # Import time for time operations
import os # Import os for file operations
from pydub import AudioSegment # Import AudioSegment for audio processing
import datetime # Import datetime for date and time operations
from scipy import signal # Import signal for signal processing
from scipy.io import wavfile # Import wavfile for audio file operations

try:
    import pyautogui # Import pyautogui for mouse control

    pyautogui.FAILSAFE = False # Set FAILSAFE to False for pyautogui
except ImportError: # If pyautogui is not available
    print("PyAutoGUI not available - mouse control features disabled") # Print message to user
    pyautogui = None # Set pyautogui to None if it is not available

# Global variables
listening = False # Set listening to False
SILENCE_THRESHOLD = 500 # Set SILENCE_THRESHOLD to 500
SAMPLE_RATE = 48000 # Set SAMPLE_RATE to 48000

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = "voice_recordings" # Set RECORDINGS_DIR to "voice_recordings"
if not os.path.exists(RECORDINGS_DIR): # If the directory does not exist
    os.makedirs(RECORDINGS_DIR) # Create the directory

print("Loading Whisper model...") # Print message to user
model = WhisperModel("small.en", device="cpu", compute_type="int8") # Load the Whisper model
print("Model loaded!") # Print message to user


def is_speech(audio_data):
    return np.max(np.abs(audio_data)) > SILENCE_THRESHOLD # Return True if the audio data is above the SILENCE_THRESHOLD


def preprocess_audio(audio_data, sample_rate=16000): # Preprocess the audio data
    try:
        # Convert to float32 for processing
        audio_float = audio_data.astype(np.float32) / 32768.0 # Convert the audio data to float32

        # Amplify the audio (adjust multiplier as needed)
        audio_float = audio_float * 2 # Amplify the audio

        # Clip to prevent distortion
        audio_float = np.clip(audio_float, -1.0, 1.0) # Clip the audio to prevent distortion

        # Convert back to int16
        audio_processed = (audio_float * 32767).astype(np.int16) # Convert the audio back to int16
        return audio_processed # Return the processed audio

    except Exception as e:
        print(f"Error in audio preprocessing: {e}") # Print message to user
        return audio_data # Return the original audio data


def save_recording_as_mp3(audio_data, recognized_text): # Save the audio recording as an MP3 file and the recognized text as a TXT file
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Get the current timestamp
        wav_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav") # Create the WAV file path
        mp3_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.mp3") # Create the MP3 file path
        txt_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.txt") # Create the TXT file path

        # Make sure the directory exists
        if not os.path.exists(RECORDINGS_DIR): # If the directory does not exist
            os.makedirs(RECORDINGS_DIR) # Create the directory

        # Save WAV file and verify it was created
        wavfile.write(wav_path, SAMPLE_RATE, audio_data) # Write the WAV file
        if not os.path.exists(wav_path): # If the WAV file does not exist
            print(f"Failed to create WAV file: {wav_path}") # Print message to user
            return None # Return None

        # Convert to MP3 only if WAV exists
        if os.path.exists(wav_path): # If the WAV file exists
            audio = AudioSegment.from_wav(wav_path) # Convert the WAV file to an AudioSegment
            audio.export(mp3_path, format="mp3", bitrate="320k") # Export the AudioSegment to an MP3 file

            # Save text file
            with open(txt_path, 'w', encoding='utf-8') as f: # Open the TXT file
                f.write(recognized_text) # Write the recognized text to the TXT file

            # Only remove WAV file if MP3 was created successfully
            if os.path.exists(mp3_path): # If the MP3 file exists
                os.remove(wav_path) # Remove the WAV file

            return mp3_path # Return the MP3 file path
        else:
            print(f"WAV file not found: {wav_path}") # Print message to user
            return None # Return None

    except Exception as e:
        print(f"Error saving recording: {e}") # Print message to user
        # If WAV file exists but conversion failed, clean up
        if os.path.exists(wav_path): # If the WAV file exists
            try:
                os.remove(wav_path) # Remove the WAV file
            except:
                pass # Do nothing
        return None # Return None


def process_voice_command(command): # Process the voice command
    command = command.lower().strip() # Convert the command to lowercase and strip whitespace
    print(f"Processing command: {command}") # Print message to user

    # List of known commands and their variations
    move_mouse_commands = ["move mouse", "move the mouse"] # List of move mouse commands
    exit_commands = ["exit window", "close window"] # List of exit commands         

    try: # Try to process the command
        # Check for move mouse commands
        if any(cmd in command for cmd in move_mouse_commands): # If the command is a move mouse command
            if "top right" in command: # If the command is to move the mouse to the top right
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right")) # Update the status label
                screen_width, _ = pyautogui.size() # Get the screen width
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5) # Move the mouse to the top right
                return True # Return True
            else: # If the command is not to move the mouse to the top right
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position")) # Update the status label
                icon_x, icon_y = 200, 200 # Set the icon position
                pyautogui.moveTo(icon_x, icon_y, duration=0.5) # Move the mouse to the icon position
                return True # Return True

        # Check for exit commands
        if any(cmd in command for cmd in exit_commands): # If the command is an exit command
            status_label.after(0, lambda: status_label.config(text="Exiting current window")) # Update the status label
            pyautogui.hotkey("alt", "f4") # Press the alt and f4 keys
            return True # Return True

        return False # Return False

    except Exception as e:
        print(f"Error in command processing: {e}") # Print message to user  
        status_label.after(0, lambda: status_label.config(text=f"Command error: {str(e)}")) # Update the status label
        return False # Return False


def save_and_process_audio(audio_data): # Save and process the audio
    try:
        # Preprocess audio for recognition
        processed_audio = preprocess_audio(audio_data) # Preprocess the audio

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file: # Create a temporary file
            temp_filename = temp_audio_file.name # Get the temporary file name
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio) # Write the processed audio to the temporary file

        print("Processing audio with Whisper...") # Print message to user
        segments, _ = model.transcribe( # Transcribe the audio
            temp_filename,  # The temporary file name
            beam_size=5, # Set the beam size
            language="en", # Set the language
            condition_on_previous_text=True, # Set the condition on previous text
            no_speech_threshold=0.3 # Set the no speech threshold
        )
        text = " ".join([segment.text for segment in segments]) # Join the segments into a single text

        if text.strip(): # If the text is not empty
            print(f"Recognized text: {text}") # Print message to user
            status_label.after(0, lambda: status_label.config(text=f"You said: {text}")) # Update the status label

            # Save recording
            mp3_path = save_recording_as_mp3(processed_audio, text) # Save the audio recording as an MP3 file and the recognized text as a TXT file

            # Process command
            if not process_voice_command(text): # If the command is not recognized
                print("Command not recognized") # Print message to user
                status_label.after(0, lambda: status_label.config(text=f"Command not recognized: {text}")) # Update the status label
        else:
            print("No speech detected") # Print message to user
            status_label.after(0, lambda: status_label.config(text="No speech detected")) # Update the status label

        os.unlink(temp_filename) # Remove the temporary file

    except Exception as e:
        print(f"Error in audio processing: {e}") # Print message to user    
        status_label.after(0, lambda: status_label.config(text=f"Processing error: {str(e)}")) # Update the status label


class AudioProcessor:
    def __init__(self):
        self.audio_buffer = [] # Initialize the audio buffer    
        self.is_recording = False # Initialize the recording flag
        self.silence_count = 0 # Initialize the silence count
        self.energy_threshold = SILENCE_THRESHOLD # Initialize the energy threshold

    def process_audio(self): # Process the audio
        CHUNK = 2048 # Set the chunk size
        MAX_SILENCE_CHUNKS = 8 # Set the maximum silence chunks

        def audio_callback(indata, frames, time_info, status): # Audio callback
            if status: # If the status is not None
                print(f"Audio callback status: {status}") # Print message to user
                return # Return None

            if listening: # If listening is True
                current_audio = indata.copy().flatten() # Get the current audio
                current_energy = np.max(np.abs(current_audio)) # Get the current energy

                if current_energy > self.energy_threshold: # If the current energy is greater than the energy threshold
                    if not self.is_recording: # If the recording is not active
                        print("Speech detected!") # Print message to user
                        self.is_recording = True # Set the recording flag to True
                        self.silence_count = 0 # Set the silence count to 0
                        self.audio_buffer.append(current_audio) # Append the current audio to the audio buffer
                elif self.is_recording: # If the recording is active
                    self.silence_count += 1 # Increment the silence count
                    self.audio_buffer.append(current_audio) # Append the current audio to the audio buffer

                    if self.silence_count >= MAX_SILENCE_CHUNKS: # If the silence count is greater than the maximum silence chunks
                        if len(self.audio_buffer) > 0: # If the audio buffer is not empty
                            complete_audio = np.concatenate(self.audio_buffer) # Concatenate the audio buffer
                            print("Processing recorded audio...") # Print message to user
                            threading.Thread(target=save_and_process_audio,
                                             args=(complete_audio,)).start() # Start the save and process audio thread

                        self.audio_buffer = [] # Set the audio buffer to an empty list
                        self.is_recording = False # Set the recording flag to False
                        self.silence_count = 0 # Set the silence count to 0

        try: # Try to start the audio stream
            print("Starting audio stream...") # Print message to user
            with sd.InputStream(callback=audio_callback, # Start the audio stream
                                channels=1, # Set the number of channels
                                samplerate=SAMPLE_RATE, # Set the sample rate
                                blocksize=CHUNK, # Set the block size
                                dtype=np.int16, # Set the data type
                                device=None,  # Use default device
                                latency='low') as stream:  # Added low latency
                print("Audio stream started") # Print message to user
                while listening: # While listening is True  
                    sd.sleep(100) # Sleep for 100 milliseconds

        except Exception as e: # If an error occurs
            print(f"Error in audio recording: {e}") # Print message to user
            status_label.after(0, lambda: status_label.config(text=f"Recording error: {str(e)}")) # Update the status label


def toggle_record(): # Toggle the recording
    global listening # Set listening to the global listening variable

    if not listening: # If listening is False
        try: # Try to start the recording
            listening = True # Set listening to True
            status_label.config(text="Listening... (Speak now)") # Update the status label
            record_button.config(text="Stop Recording") # Update the record button text

            audio_processor = AudioProcessor() # Create an AudioProcessor object
            recording_thread = threading.Thread(target=audio_processor.process_audio) # Create a recording thread
            recording_thread.daemon = True # Set the recording thread to daemon
            recording_thread.start() # Start the recording thread
            print("Recording thread started") # Print message to user

        except Exception as e: # If an error occurs
            print(f"Error starting recording: {e}") # Print message to user
            status_label.config(text=f"Error: {str(e)}") # Update the status label
            record_button.config(text="Record") # Update the record button text
            listening = False # Set listening to False

    else: # If listening is True    
        listening = False # Set listening to False
        record_button.config(text="Record") # Update the record button text
        status_label.config(text="Stopped listening") # Update the status label
        print("Stopped listening") # Print message to user


# Set up the main Tkinter window
root = tk.Tk() # Create a Tkinter window    
root.title("Voice to Text Demo") # Set the window title
root.geometry("400x200") # Set the window size

# Add a label to display instructions or results
status_label = tk.Label(root, text="Press the button and speak.", wraplength=300) # Create a status label
status_label.pack(pady=20) # Add the status label to the window

# Add a button to start/stop recording
record_button = tk.Button(root, text="Record", command=toggle_record) # Create a record button
record_button.pack() # Add the record button to the window

root.mainloop() # Start the Tkinter event loop
