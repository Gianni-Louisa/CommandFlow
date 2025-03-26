"""
# This is the module docstring that provides overall information about the file
Code Artifact: cudaToText.py
Description: Program to listen for audio input and handle commands

@author: Gianni Louisa, Connor Bennudriti, Ethan Dirkes, Christoper Gronewold, Tommy Lam
@created: 2/14/2025
@revised: 3/2/2025

Revision History:
- 2/14/2025: Initial creation of script
- 3/2/2025 (Ethan Dirkes): Added image to record button and label to display detected speech
- 3/2/2025: Commented code
- 3/12/2025: Adjusted GUI and adjusted model

Preconditions:
- OpenAI's whisper library must be installed
- Program must be run on Nvidia GPU with CUDA in order for the CUDA functionality to work (but is not needed for code execution)

Postconditions:
- Displays audio on a GUI
- Performs actions that are commanded by the user in the audio

"""

import tkinter as tk  # Import Tkinter library for creating GUI elements and windows
from faster_whisper import WhisperModel  # Import WhisperModel class from faster_whisper for speech recognition
import sounddevice as sd  # Import sounddevice for audio recording and playback
import numpy as np  # Import numpy for numerical operations and array handling
import threading  # Import threading module to handle concurrent execution
import tempfile  # Import tempfile module to create temporary files
import os  # Import os module for operating system dependent functionality
from scipy.io import wavfile  # Import wavfile module from scipy.io for reading and writing WAV files
import queue  # Import queue for thread-safe data exchange
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for managing thread pools for background tasks
import torch  # Import PyTorch to check for CUDA availability and GPU support
from window_detection import get_window_snapshot, get_context_for_speech_command  # Import custom functions for window detection and context analysis

try:
    import pyautogui  # Try to import pyautogui module for mouse and keyboard control
    pyautogui.FAILSAFE = False  # Disable the failsafe feature that stops mouse movement when cursor hits screen corner
except ImportError:
    print("PyAutoGUI not available - mouse control features disabled")  # Print a message indicating mouse control features are disabled
    pyautogui = None  # Set pyautogui to None so we can check if it's available later

SILENCE_THRESHOLD = 500  # Define the energy threshold to determine when speech is occurring
SAMPLE_RATE = 48000  # Define the audio sampling rate in Hz (48kHz is high quality audio)

listening_event = threading.Event()  # Create a threading event to control when the app is actively listening

executor = ThreadPoolExecutor(max_workers=4)  # Create a thread pool with 4 workers to handle background processing tasks

device = "cuda" if torch.cuda.is_available() else "cpu"  # Determine whether to use GPU or CPU for processing based on CUDA availability

print("Loading Whisper model...")  # Print a message indicating that the Whisper model is being loaded
model = WhisperModel("small.en", device=device, compute_type="int8")  # Initialize the Whisper speech recognition model with English language, small size, and int8 quantization
print("Model loaded!")  # Print a message indicating that the model has been loaded successfully


def preprocess_audio(audio_data, sample_rate=16000):
    """
    # Function docstring describing what preprocess_audio does
    proprocess_audio: Function to process audio so it is more clear
    """
    try:
        audio_float = audio_data.astype(np.float32) / 32768.0  # Convert the audio data from int16 to float32 and normalize to range [-1, 1]
        audio_float *= 2  # Amplify the signal by multiplying by 2
        np.clip(audio_float, -1.0, 1.0, out=audio_float)  # Clip values to prevent distortion, keeping them in range [-1, 1]
        audio_processed = (audio_float * 32767).astype(np.int16)  # Convert the audio back to int16 format for compatibility
        return audio_processed  # Return the processed audio data
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")  # Print error message if any exception occurs during preprocessing
        return audio_data  # Return the original audio data if processing fails


def process_voice_command(command):
    """
    # Function docstring describing what process_voice_command does
    process_voice_command(): Function to perform the command that was heard by the audio listener
    """
    command = command.lower().strip()  # Normalize the command by converting to lowercase and removing whitespace
    print(f"Processing command: {command}")  # Print the command being processed for debugging
    move_mouse_commands = ["move mouse", "move the mouse"]  # Define list of commands related to mouse movement for easier matching
    exit_commands = ["exit window", "close window"]  # Define list of commands related to closing windows for easier matching

    try:
        # Update feedback UI to show command being processed
        update_feedback_display("Processing command...", "processing")
        
        if any(cmd in command for cmd in move_mouse_commands):  # Check if any of the mouse movement commands are in the recognized text
            if "top right" in command:  # Check if "top right" is specified in the command
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))  # Use tkinter's after method to update status label safely from another thread
                screen_width, _ = pyautogui.size()  # Get the screen width and height (only using width here)
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5)  # Move the mouse to the top-right corner of the screen over 0.5 seconds
                update_feedback_display("Command executed successfully", "success")
                return True  # Return True to indicate command was handled
            else:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position"))  # If no specific location mentioned, move to default position
                icon_x, icon_y = 200, 200  # Define default position coordinates
                pyautogui.moveTo(icon_x, icon_y, duration=0.5)  # Move the mouse to the default position over 0.5 seconds
                update_feedback_display("Command executed successfully", "success")
                return True  # Return True to indicate command was handled

        if any(cmd in command for cmd in exit_commands):  # Check if any window closing commands are in the recognized text
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))  # Update status label to show we're exiting the window
            pyautogui.hotkey("alt", "f4")  # Simulate Alt+F4 keyboard shortcut to close the active window
            update_feedback_display("Command executed successfully", "success")
            return True  # Return True to indicate command was handled

        # If we reach here, no command was recognized
        update_feedback_display("No matching command found", "error")
        return False  # Return False if no matching command was found

    except Exception as e:
        print(f"Error in command processing: {e}")  # Print error message if any exception occurs during command processing
        status_label.after(0, lambda: status_label.config(text=f"Command error: {str(e)}"))  # Update status label to show the error
        update_feedback_display("Command execution failed", "error")
        return False  # Return False to indicate command handling failed


def update_feedback_display(message, status_type):
    """
    Update the feedback display with command execution status
    
    Args:
        message (str): The feedback message to display
        status_type (str): The type of status - 'processing', 'success', or 'error'
    """
    # Define colors for different status types
    status_colors = {
        "processing": "#3498db",  # Blue
        "success": "#2ecc71",     # Green
        "error": "#e74c3c"        # Red
    }
    
    # Get the color for this status type
    color = status_colors.get(status_type, "#7f8c8d")
    
    # Update the feedback display on the GUI thread
    feedback_display.after(0, lambda: feedback_display.config(
        text=message,
        fg="white",
        bg=color
    ))
    
    # Log the feedback in console too
    print(f"Feedback ({status_type}): {message}")
    
    # Clear the feedback after a delay for success messages
    if status_type == "success":
        feedback_display.after(3000, lambda: feedback_display.config(
            text="Ready for next command",
            fg="white",
            bg="#1a2332"
        ))


def save_and_process_audio(audio_data):
    """
    # Function docstring describing what save_and_process_audio does
    save_and_process_audio(): Function to save the audio to a wav file and then process it
    """
    try:
        update_feedback_display("Processing audio...", "processing")
        processed_audio = preprocess_audio(audio_data)  # Preprocess the audio to enhance recognition quality
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:  # Create a temporary WAV file with '.wav' extension that won't be immediately deleted
            temp_filename = temp_audio_file.name  # Get the name of the temporary file
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio)  # Write the processed audio data to the temporary file

        print("Processing audio with Whisper...")  # Print status message about audio processing
        segments, _ = model.transcribe(  # Transcribe the audio using the Whisper model and get the segments and info
            temp_filename,  # Path to the audio file
            beam_size=5,  # Beam search size for more accurate transcription
            language="en",  # Force English language for recognition
            condition_on_previous_text=True,  # Use context from previous segments
            no_speech_threshold=0.3  # Threshold for filtering out non-speech
        )
        text = " ".join([segment.text for segment in segments])  # Combine all segments into a single text string
        if text.strip():  # Check if any text was recognized
            print(f"Recognized text: {text}")  # Print the recognized text for debugging
            text_input.delete("1.0", tk.END)  # Delete all text in the text input widget
            text_input.insert("1.0", text)  # Insert the recognized text into the text input widget
            update_feedback_display("Speech recognized", "success")
            context = get_context_for_speech_command(text)  # Get context to determine if this is likely a false positive

            if context.get("likely_false_positive"):  # Check if the recognition is likely a false positive based on context
                print(f"Ignoring likely false recognition: {text}")  # Log that we're ignoring a likely false recognition
                update_feedback_display("Ignored likely false recognition", "error")
            else:
                process_voice_command(text)  # Process the recognized text as a command
        else:
            print("No speech detected")  # Log that no speech was detected
            text_input.delete("1.0", tk.END)  # Clear the text input widget
            text_input.insert("1.0", "No speech detected")  # Display "No speech detected" message in the text input widget
            update_feedback_display("No speech detected", "error")

        os.unlink(temp_filename)  # Delete the temporary file to clean up

    except Exception as e:
        print(f"Error in audio processing: {e}")  # Print error message if any exception occurs during audio processing
        text_input.delete("1.0", tk.END)  # Clear the text input widget
        text_input.insert("1.0", f"Processing error: {str(e)}")  # Display the error message in the text input widget
        update_feedback_display("Audio processing error", "error")


class AudioProcessor:
    """
    # Class docstring describing what AudioProcessor does
    AudioProcessor: A class to process the user's command audio
    """
    def __init__(self):
        self.audio_queue = queue.Queue()  # Initialize a queue to store incoming audio chunks
        self.audio_buffer = []  # Initialize an empty list to accumulate audio during speech
        self.recording_active = False  # Initialize flag to track if speech is currently being recorded
        self.silence_count = 0  # Initialize counter to track consecutive silent chunks
        self.energy_threshold = SILENCE_THRESHOLD  # Set energy threshold to distinguish speech from silence

    def audio_callback(self, indata, frames, time_info, status):
        """
        # Method docstring describing what audio_callback does
        audio_callback: A callback method for the sounddevice InputStream
        """
        if status:  # Check if there's any status information to log
            print(f"Audio callback status: {status}")  # Log any issues with audio input
        if listening_event.is_set():  # Only process audio if listening is enabled
            self.audio_queue.put(indata.copy().flatten())  # Flatten the input array to 1D, make a copy, and add to queue

    def process_audio(self):
        """
        # Method docstring describing what process_audio does
        process_audio: Main class method to process the audio
        """
        CHUNK = 8192  # Define size of each audio chunk in samples
        MAX_SILENCE_CHUNKS = 8  # Define number of silent chunks to wait before processing (determines pause length)

        try:
            print("Starting audio stream...")  # Print status message about starting audio stream
            with sd.InputStream(  # Create and start the audio input stream with specified parameters
                callback=self.audio_callback,  # Set the callback function
                channels=1,  # Record in mono
                samplerate=SAMPLE_RATE,  # Set the sample rate
                blocksize=CHUNK,  # Set the block size
                dtype=np.int16,  # Use 16-bit integer samples
                latency='low'  # Use low latency for responsive detection
            ) as stream: 
                print("Audio stream started")  # Print status message that audio stream started

                while listening_event.is_set():  # Main processing loop that runs while listening is enabled
                    try:
                        current_audio = self.audio_queue.get(timeout=0.15)  # Try to get the next audio chunk from the queue with a timeout to prevent blocking
                    except queue.Empty:
                        continue  # If queue is empty, skip this iteration and try again

                    energy = np.max(np.abs(current_audio))  # Calculate audio energy (maximum absolute amplitude)

                    if energy > self.energy_threshold:  # Check if energy exceeds threshold (speech detected)
                        if not self.recording_active:  # Check if we're not already recording
                            print("Speech detected!")  # Log that speech was detected
                            self.recording_active = True  # Start recording session
                            self.silence_count = 0  # Reset silence counter
                        self.audio_buffer.append(current_audio)  # Add current audio chunk to buffer

                    elif self.recording_active:  # If we're already recording but current chunk is silent
                        self.audio_buffer.append(current_audio)  # Add silent chunk to buffer
                        self.silence_count += 1  # Increment silence counter
                        
                        if self.silence_count >= MAX_SILENCE_CHUNKS:  # Check if enough consecutive silent chunks to finish recording
                            complete_audio = np.concatenate(self.audio_buffer)  # Combine all buffered audio chunks into one array
                            print("Processing recorded audio...")  # Log that we're processing the recorded audio

                            executor.submit(save_and_process_audio, complete_audio)  # Submit the processing task to the thread pool
                            
                            self.audio_buffer = []  # Reset audio buffer to empty list
                            self.recording_active = False  # Set recording_active flag to False
                            self.silence_count = 0  # Reset silence counter to 0

        except Exception as e:
            print(f"Error in audio recording: {e}")  # Print error message if any exception occurs during audio recording
            text_input.delete("1.0", tk.END)  # Clear the text input widget
            text_input.insert("1.0", f"Recording error: {str(e)}")  # Display the error message in the text input widget


def toggle_record():
    """
    # Function docstring describing what toggle_record does
    toggle_record(): Function to turn audio recording off/on
    """
    if not listening_event.is_set():  # Check if listening is not currently enabled
        try:
            listening_event.set()  # Enable listening by setting the event
            
            # Update feedback display
            update_feedback_display("Listening for commands...", "processing")
            
            audio_processor = AudioProcessor()  # Create an AudioProcessor instance

            recording_thread = threading.Thread(target=audio_processor.process_audio, daemon=True)  # Create a new thread for audio processing that will run in the background
            recording_thread.start()  # Start the recording thread
            print("Recording thread started")  # Log that recording thread started

        except Exception as e:
            print(f"Error starting recording: {e}")  # Print error message if any exception occurs when starting recording
            status_label.config(text=f"Error: {str(e)}")  # Update status label with the error message
            update_feedback_display(f"Recording error: {str(e)}", "error")
            listening_event.clear()  # Clear the listening event to stop audio processing
    else:
        listening_event.clear()  # Stop listening by clearing the event
        status_label.config(text="Press button and speak")  # Update status label to show stopped state
        text_input.delete("1.0", tk.END)  # Clear the text input widget
        text_input.insert("1.0", "Stopped listening")  # Display "Stopped listening" message in the text input widget
        update_feedback_display("Recording stopped", "success")
        print("Stopped listening")  # Log that listening stopped


root = tk.Tk()  # Create the main Tkinter window
root.title("CommandFlow")  # Set the window title
root.geometry("800x600")  # Increased height to accommodate all elements
root.configure(bg="#212a38")  # Set the background color to dark blue

main_container = tk.Frame(root, bg="#212a38", padx=0, pady=0)  # Create main container frame with no padding
main_container.pack(fill=tk.BOTH, expand=True)  # Pack the main container to fill the window

sidebar = tk.Frame(main_container, width=340, bg="#ffffff", padx=0, pady=0)  # Create sidebar frame with white background
sidebar.pack(side=tk.LEFT, fill=tk.Y)  # Pack the sidebar on the left side
sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking

sidebar_content = tk.Frame(sidebar, bg="#ffffff", padx=25, pady=30)  # Add padding container inside sidebar for content
sidebar_content.pack(fill=tk.BOTH, expand=True)  # Pack the sidebar content to fill the sidebar

app_title = tk.Label(sidebar_content, text="CommandFlow", font=("Segoe UI", 22, "bold"),   # Create app title label with modern typography
                    bg="#ffffff", fg="#212a38")
app_title.pack(anchor=tk.W, pady=(0, 40))  # Pack the app title at the top of the sidebar with padding

mic_image = tk.PhotoImage(file="GUI Resources/mic-icon.png")  # Load microphone icon image
record_button = tk.Button(sidebar_content, image=mic_image, text="",   # Create button with the microphone image
                         compound=tk.CENTER, bd=0, bg="#ffffff", 
                         activebackground="#ffffff", command=toggle_record,
                         cursor="hand2", highlightthickness=0)
record_button.pack(pady=(0, 30))  # Pack the record button with padding

status_label = tk.Label(sidebar_content, text="Press to speak",  # Create status label with instructions
                       font=("Segoe UI", 12), bg="#ffffff", fg="#4a5568")
status_label.pack(pady=(0, 20))  # Pack the status label with padding

content_area = tk.Frame(main_container, bg="#212a38", padx=40, pady=40)  # Create main content area with blue background
content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Pack the content area on the right side

content_title = tk.Label(content_area, text="Voice Recognition",   # Add title to content area
                        font=("Segoe UI", 18, "bold"), bg="#212a38", fg="#ffffff")
content_title.pack(anchor=tk.W, pady=(0, 30))  # Pack the content title at the top of the content area with padding

# Create transcript container with fixed height to prevent it from taking too much space
transcript_frame = tk.Frame(content_area, bg="#1a2332", bd=0, height=250)  # Set fixed height
transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))  # Pack the transcript frame
transcript_frame.pack_propagate(False)  # Prevent the frame from shrinking to fit its contents

transcript_header = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=20)  # Add transcript header with styling
transcript_header.pack(fill=tk.X)  # Pack the transcript header to fill horizontally

transcript_title = tk.Label(transcript_header, text="Transcription",   # Create label for transcript title
                          font=("Segoe UI", 14), bg="#1a2332", fg="#ffffff")
transcript_title.pack(anchor=tk.W)  # Pack the transcript title on the left side

separator = tk.Frame(transcript_frame, height=1, bg="#2c3445")  # Create a subtle separator line
separator.pack(fill=tk.X)  # Pack the separator to fill horizontally

transcript_content = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=25)  # Create frame for transcript content with padding
transcript_content.pack(fill=tk.BOTH, expand=True)  # Pack the transcript content to fill the transcript frame

text_input = tk.Text(transcript_content,   # Create a Text widget for input and display
                   wrap=tk.WORD,  # Wrap text by words
                   fg="#b3c0d1",  # Light blue text color
                   bg="#1a2332",  # Dark blue background
                   font=("Segoe UI", 12),  # Modern font
                   bd=0,  # No border
                   padx=0,  # No horizontal padding
                   pady=0,  # No vertical padding
                   insertbackground="#ffffff",  # White cursor color
                   selectbackground="#3a4555",  # Selection background color
                   selectforeground="#ffffff",  # Selection text color
                   highlightthickness=0)  # No focus highlight
text_input.pack(fill=tk.BOTH, expand=True)  # Pack the text input to fill the transcript content area
text_input.insert("1.0", "Type or speak your command here...")  # Insert placeholder text

def on_focus_in(event):  # Define function for focus-in event
    if text_input.get("1.0", "end-1c") == "Type or speak your command here...":  # Check if text contains the placeholder
        text_input.delete("1.0", tk.END)  # Clear the placeholder text
        
def on_focus_out(event):  # Define function for focus-out event
    if text_input.get("1.0", "end-1c").strip() == "":  # Check if text is empty
        text_input.insert("1.0", "Type or speak your command here...")  # Insert the placeholder text

text_input.bind("<FocusIn>", on_focus_in)  # Bind focus-in event to the function
text_input.bind("<FocusOut>", on_focus_out)  # Bind focus-out event to the function

def process_typed_command(event):  # Define function to process commands when Enter is pressed
    command = text_input.get("1.0", "end-1c").strip()  # Get the text from the input widget
    if command and command != "Type or speak your command here...":  # Check if there's a command and it's not the placeholder
        process_voice_command(command)  # Process the command using the same function for spoken commands
        text_input.delete("1.0", tk.END)  # Clear the input after processing
    return "break"  # Return "break" to prevent default Enter behavior

text_input.bind("<Return>", process_typed_command)  # Bind Enter key press to the function

snapshot = get_window_snapshot()  # Get a snapshot of all open windows

all_open_windows = snapshot["all_windows"]  # Get the complete list of all open windows

for window in all_open_windows:  # Loop through all windows and print their titles and application names
    print(f"Window: {window.get('title')} - Application: {window.get('app_name')}")

active_app = snapshot["active_window"]["app_name"]  # Get the currently active application name
print(f"You're currently using: {active_app}")  # Print the currently active application

# Create a feedback section with more visibility
feedback_frame = tk.Frame(content_area, bg="#1a2332", bd=0)
feedback_frame.pack(fill=tk.X, expand=False, pady=(0, 15))  # Ensure it's packed with padding

feedback_header = tk.Frame(feedback_frame, bg="#1a2332", padx=25, pady=15)
feedback_header.pack(fill=tk.X)

feedback_title = tk.Label(feedback_header, text="Command Status", 
                         font=("Segoe UI", 14, "bold"), bg="#1a2332", fg="#ffffff")  # Make the title bold
feedback_title.pack(anchor=tk.W)

separator_feedback = tk.Frame(feedback_frame, height=1, bg="#2c3445")
separator_feedback.pack(fill=tk.X)

# Increase the height of the feedback content area
feedback_content = tk.Frame(feedback_frame, bg="#1a2332", padx=25, pady=20, height=80)  # Increased height and padding
feedback_content.pack(fill=tk.X)
feedback_content.pack_propagate(False)  # Prevent shrinking

# Make the colored background larger with more padding
feedback_display = tk.Label(feedback_content, 
                          text="Ready for commands", 
                          font=("Segoe UI", 12),
                          bg="#1a2332", 
                          fg="white",
                          anchor=tk.CENTER,  # Center the text
                          padx=20,          # More horizontal padding
                          pady=15,          # More vertical padding
                          wraplength=400,
                          justify=tk.CENTER) # Center-justify the text
feedback_display.pack(fill=tk.BOTH, expand=True)  # Fill both directions and expand

root.mainloop()  # Start the Tkinter main event loop
