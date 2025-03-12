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
from window_detection import get_window_snapshot, get_context_for_speech_command

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
            text_input.delete("1.0", tk.END)
            text_input.insert("1.0", text)
            # Get context to determine if this is likely a false positive
            context = get_context_for_speech_command(text)

            if context.get("likely_false_positive"):
                print(f"Ignoring likely false recognition: {text}")
            else:
                # Process as normal command
                process_voice_command(text)
        # If no text was recognized
        else:
            print("No speech detected")
            text_input.delete("1.0", tk.END)
            text_input.insert("1.0", "No speech detected") # display an error message

        # Clean up temporary file
        os.unlink(temp_filename)

    # Handle error by displaying info
    except Exception as e:
        print(f"Error in audio processing: {e}")
        text_input.delete("1.0", tk.END)
        text_input.insert("1.0", f"Processing error: {str(e)}")


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
            text_input.delete("1.0", tk.END)
            text_input.insert("1.0", f"Recording error: {str(e)}")


def toggle_record():
    """
    toggle_record(): Function to turn audio recording off/on
    """
    if not listening_event.is_set():
        try:
            listening_event.set() # Start listening
            
            # Update UI to show listening state
      

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
            listening_event.clear()
    else:
        # Stop listening
        listening_event.clear() # Clear the event to stop audio processing
        # Update UI to show stopped state
        status_label.config(text="Press button and speak")
        text_input.delete("1.0", tk.END)
        text_input.insert("1.0", "Stopped listening")
        print("Stopped listening")


# Set up the main Tkinter window
root = tk.Tk()
root.title("CommandFlow")
root.geometry("800x500")  # Larger size for better proportions
root.configure(bg="#212a38")  # Blue background for the root

# Create main container with padding
main_container = tk.Frame(root, bg="#212a38", padx=0, pady=0)
main_container.pack(fill=tk.BOTH, expand=True)

# Create sidebar frame - now white to match with photo background
sidebar = tk.Frame(main_container, width=340, bg="#ffffff", padx=0, pady=0)
sidebar.pack(side=tk.LEFT, fill=tk.Y)
sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking

# Add padding container inside sidebar for content
sidebar_content = tk.Frame(sidebar, bg="#ffffff", padx=25, pady=30)
sidebar_content.pack(fill=tk.BOTH, expand=True)

# Create app title with modern typography (now dark text on white background)
app_title = tk.Label(sidebar_content, text="CommandFlow", font=("Segoe UI", 22, "bold"), 
                    bg="#ffffff", fg="#212a38")
app_title.pack(anchor=tk.W, pady=(0, 40))

# Load and display mic image directly without restrictions
mic_image = tk.PhotoImage(file="GUI Resources/mic-icon.png")
record_button = tk.Button(sidebar_content, image=mic_image, text="", 
                         compound=tk.CENTER, bd=0, bg="#ffffff", 
                         activebackground="#ffffff", command=toggle_record,
                         cursor="hand2", highlightthickness=0)
record_button.pack(pady=(0, 30))

# Status text with updated styling (dark text on white background)
status_label = tk.Label(sidebar_content, text="Press to speak",
                       font=("Segoe UI", 12), bg="#ffffff", fg="#4a5568")
status_label.pack(pady=(0, 20))

# Create main content area - now blue
content_area = tk.Frame(main_container, bg="#212a38", padx=40, pady=40)
content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Add minimalist title to content area (white text on blue background)
content_title = tk.Label(content_area, text="Voice Recognition", 
                        font=("Segoe UI", 18, "bold"), bg="#212a38", fg="#ffffff")
content_title.pack(anchor=tk.W, pady=(0, 30))

# Create a darker blue frame for the transcription
transcript_frame = tk.Frame(content_area, bg="#1a2332", bd=0)
transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

# Add transcript header with updated styling
transcript_header = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=20)
transcript_header.pack(fill=tk.X)

transcript_title = tk.Label(transcript_header, text="Transcription", 
                          font=("Segoe UI", 14), bg="#1a2332", fg="#ffffff")
transcript_title.pack(anchor=tk.W)

# Subtle separator - slightly lighter blue
separator = tk.Frame(transcript_frame, height=1, bg="#2c3445")
separator.pack(fill=tk.X)

# Transcript content area with better spacing
transcript_content = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=25)
transcript_content.pack(fill=tk.BOTH, expand=True)

# Replace the label with a typable Text widget
text_input = tk.Text(transcript_content, 
                   wrap=tk.WORD,
                   fg="#b3c0d1", 
                   bg="#1a2332",
                   font=("Segoe UI", 12),
                   bd=0,  # No border
                   padx=0,
                   pady=0,
                   insertbackground="#ffffff",  # White cursor
                   selectbackground="#3a4555",  # Selection background
                   selectforeground="#ffffff",  # Selection text color
                   highlightthickness=0)  # No focus highlight
text_input.pack(fill=tk.BOTH, expand=True)
text_input.insert("1.0", "Type or speak your command here...")

# Add focus in/out effects for better UX
def on_focus_in(event):
    if text_input.get("1.0", "end-1c") == "Type or speak your command here...":
        text_input.delete("1.0", tk.END)
        
def on_focus_out(event):
    if text_input.get("1.0", "end-1c").strip() == "":
        text_input.insert("1.0", "Type or speak your command here...")

text_input.bind("<FocusIn>", on_focus_in)
text_input.bind("<FocusOut>", on_focus_out)

# Optional: Add a function to process typed commands when user presses Enter
def process_typed_command(event):
    command = text_input.get("1.0", "end-1c").strip()
    if command and command != "Type or speak your command here...":
        # Process the command (same logic as for spoken commands)
        # This can call the same function that processes recognized speech
        process_voice_command(command)
        # Optionally clear the input after processing
        text_input.delete("1.0", tk.END)
    return "break"  # Prevents default Enter behavior

text_input.bind("<Return>", process_typed_command)

# Get the snapshot of ALL open windows
snapshot = get_window_snapshot()

# Access the complete list of ALL windows
all_open_windows = snapshot["all_windows"]  

# Print all window titles
for window in all_open_windows:
    print(f"Window: {window.get('title')} - Application: {window.get('app_name')}")

# Use the information
active_app = snapshot["active_window"]["app_name"]
print(f"You're currently using: {active_app}")
# Start the main application loop
root.mainloop()
