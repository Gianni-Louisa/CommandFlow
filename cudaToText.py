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
- 3/30/2025: Added task automation and moved mouse commands

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
import time  # Import time module for task tracking
import subprocess  # Import subprocess module for running external scripts
import webbrowser  # Import webbrowser module for opening websites

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

# Global variables for tracking active tasks
active_tasks = {}  # Dictionary to store active tasks and their status
task_id_counter = 0  # Counter for generating unique task IDs

# Function to update the task status display
def update_task_status_display():
    """
    Update the task status display to show currently active tasks
    """
    # Clear the current display
    for widget in task_status_content.winfo_children():
        widget.destroy()
        
    # Check if there are any active tasks
    if not active_tasks:
        no_tasks_label = tk.Label(
            task_status_content,
            text="No active tasks",
            font=("Segoe UI", 10),
            bg="#1a2332",
            fg="#7f8c8d",
            anchor=tk.W,
            padx=5,
            pady=3
        )
        no_tasks_label.pack(fill=tk.X, padx=5, pady=2)
        return
        
    # Add a label for each active task
    for task_id, task_info in active_tasks.items():
        task_frame = tk.Frame(task_status_content, bg="#1a2332", padx=0, pady=0)
        task_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Status color based on status type
        status_colors = {
            "processing": "#3498db",  # Blue
            "success": "#2ecc71",     # Green
            "error": "#e74c3c",       # Red
            "completed": "#f39c12"    # Orange (for completed tasks waiting to be cleared)
        }
        
        # Status indicator (colored dot)
        status_indicator = tk.Label(
            task_frame,
            text="●",
            font=("Segoe UI", 12),
            bg="#1a2332",
            fg=status_colors.get(task_info["status"], "#7f8c8d"),
            width=2,
            anchor=tk.W
        )
        status_indicator.pack(side=tk.LEFT)
        
        # Task description label
        task_label = tk.Label(
            task_frame,
            text=task_info["description"][:40] + "..." if len(task_info["description"]) > 40 else task_info["description"],
            font=("Segoe UI", 10),
            bg="#1a2332",
            fg="white",
            anchor=tk.W
        )
        task_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add tooltip with full task description
        def show_tooltip(event, text=task_info["description"]):
            tooltip = tk.Toplevel(root)
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip_label = tk.Label(tooltip, text=text, justify=tk.LEFT, 
                                    background="#1a2332", foreground="white", 
                                    relief=tk.SOLID, borderwidth=1, padx=5, pady=2)
            tooltip_label.pack()
            
            def hide_tooltip(_):
                tooltip.destroy()
                
            task_label.bind("<Leave>", hide_tooltip)
            
        task_label.bind("<Enter>", show_tooltip)
        
        # Status label
        status_text = task_info.get("status_message", task_info["status"].capitalize())
        status_label = tk.Label(
            task_frame,
            text=status_text,
            font=("Segoe UI", 10),
            bg="#1a2332",
            fg=status_colors.get(task_info["status"], "#7f8c8d"),
            anchor=tk.E,
            width=12
        )
        status_label.pack(side=tk.RIGHT)

# Function to add or update a task in the task tracker
def track_task(description, status="processing", status_message=None):
    """
    Add or update a task in the task tracker
    
    Args:
        description (str): Description of the task
        status (str): Status of the task - 'processing', 'success', 'error', or 'completed'
        status_message (str, optional): Optional status message to display
    
    Returns:
        int: The task ID assigned to this task
    """
    global task_id_counter
    
    # Check if this task already exists (by description)
    existing_task_id = None
    for task_id, task_info in active_tasks.items():
        if task_info["description"] == description:
            existing_task_id = task_id
            break
    
    if existing_task_id is not None:
        # Update existing task
        active_tasks[existing_task_id]["status"] = status
        if status_message:
            active_tasks[existing_task_id]["status_message"] = status_message
        task_id = existing_task_id
    else:
        # Create new task
        task_id = task_id_counter
        task_id_counter += 1
        active_tasks[task_id] = {
            "description": description,
            "status": status,
            "start_time": time.time(),
            "status_message": status_message or status.capitalize()
        }
    
    # Update the task status display
    feedback_display.after(0, update_task_status_display)
    
    return task_id

# Function to remove a task from the tracker
def remove_task(task_id):
    """
    Remove a task from the task tracker
    
    Args:
        task_id (int): The ID of the task to remove
    """
    if task_id in active_tasks:
        del active_tasks[task_id]
        feedback_display.after(0, update_task_status_display)

# Function to clear completed tasks
def clear_completed_tasks():
    """
    Remove all completed tasks from the task tracker
    """
    completed_task_ids = [task_id for task_id, task_info in active_tasks.items() 
                         if task_info["status"] in ["success", "completed", "error"]]
    
    for task_id in completed_task_ids:
        remove_task(task_id)
    
    # Also schedule a periodic cleanup for any tasks that are older than 5 minutes
    current_time = time.time()
    stale_task_ids = [task_id for task_id, task_info in active_tasks.items() 
                     if current_time - task_info.get("start_time", current_time) > 300]  # 5 minutes
    
    for task_id in stale_task_ids:
        remove_task(task_id)

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


def process_voice_command(command_text):
    """
    Process a voice command from the text input
    
    Args:
        command_text (str): The text of the command to process
    """
    # Normalize the command text (lowercase, remove extra spaces, etc.)
    command_text = command_text.lower().strip()
    
    print(f"DEBUG: Processing voice command: '{command_text}'")
    
    try:
        # If there's no actual command, just return
        if not command_text:
            return
            
        # Cancel any pending clear operations first to avoid race conditions
        for after_id in feedback_display.tk.call('after', 'info'):
            try:
                feedback_display.after_cancel(int(after_id))
                print(f"DEBUG: Cancelled after task with ID {after_id}")
            except ValueError:
                pass  # Not a numeric ID
                
        # Show a processing message - make sure it won't auto-clear
        msg = "Processing command..."
        print(f"DEBUG: Setting processing message: '{msg}'")
        update_feedback_display(msg, "processing", auto_clear=False)
        
        # Check for different command types and execute the corresponding action
        
        # Task automation command - starts with 'task:'
        if command_text.startswith('task:'):
            task_description = command_text[5:].strip()
            if task_description:
                print(f"DEBUG: Detected task automation command: '{task_description}'")
                # Handle task automation in a separate thread to avoid blocking the GUI
                handle_task_automation(task_description)
            else:
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):
                    try:
                        feedback_display.after_cancel(int(after_id))
                    except ValueError:
                        pass  # Not a numeric ID
                
                error_msg = "No task description provided. Please specify a task after 'task:'"
                print(f"DEBUG: {error_msg}")
                update_feedback_display(error_msg, "error")
                
        # Open website command - starts with 'open:'
        elif command_text.startswith('open:'):
            website = command_text[5:].strip()
            if website:
                # Track this command
                task_id = track_task(f"Open website: {website}", "processing")
                
                print(f"DEBUG: Opening website: {website}")
                if not website.startswith(('http://', 'https://')):
                    website = 'https://' + website
                # Open the website in the default browser
                webbrowser.open(website)
                
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):
                    try:
                        feedback_display.after_cancel(int(after_id))
                    except ValueError:
                        pass  # Not a numeric ID
                
                success_msg = f"Opening {website}"
                print(f"DEBUG: {success_msg}")
                update_feedback_display(success_msg, "success")
                
                # Update task status
                track_task(f"Open website: {website}", "success", "Opened")
                
                # Schedule task removal after a delay
                feedback_display.after(5000, lambda: remove_task(task_id))
                
            else:
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):
                    try:
                        feedback_display.after_cancel(int(after_id))
                    except ValueError:
                        pass  # Not a numeric ID
                
                error_msg = "No website provided. Please specify a website after 'open:'"
                print(f"DEBUG: {error_msg}")
                update_feedback_display(error_msg, "error")
                
        # Move mouse command
        elif "move mouse" in command_text or "move the mouse" in command_text:
            # Track this command
            task_id = track_task("Move mouse", "processing")
            
            if "top right" in command_text:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))
                screen_width, _ = pyautogui.size()
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5)
                
                # Update task status
                track_task("Move mouse", "success", "Top Right")
                
                update_feedback_display("Mouse moved to top right", "success")
            else:
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default position"))
                icon_x, icon_y = 200, 200
                pyautogui.moveTo(icon_x, icon_y, duration=0.5)
                
                # Update task status
                track_task("Move mouse", "success", "Default Pos")
                
                update_feedback_display("Mouse moved to default position", "success")
                
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))
            
        # Close window command
        elif "exit window" in command_text or "close window" in command_text:
            # Track this command
            task_id = track_task("Close window", "processing")
            
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))
            pyautogui.hotkey("alt", "f4")
            
            # Update task status
            track_task("Close window", "success", "Closed")
            
            update_feedback_display("Window closed", "success")
            
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))
            
        # Command not recognized
        else:
            # If none of the direct commands matched, try task automation
            print(f"DEBUG: No direct command match, attempting task automation for: '{command_text}'")
            handle_task_automation(command_text)
            
    except Exception as e:
        print(f"Error processing command: {e}")
        # Cancel any pending clear operations
        for after_id in feedback_display.tk.call('after', 'info'):
            try:
                feedback_display.after_cancel(int(after_id))
            except ValueError:
                pass  # Not a numeric ID
        
        error_msg = f"Error processing command: {str(e)}"
        print(f"DEBUG: {error_msg}")
        update_feedback_display(error_msg, "error")


def launch_task_automation(task_description):
    """
    Launch the task automation with the given task description
    
    Args:
        task_description (str): The description of the task to automate
    """
    # Add the task to the tracker
    task_id = track_task(f"Task: {task_description}", "processing", "Starting...")
    
    # This function is already being called in a separate thread by executor.submit(),
    # so we don't need to create another thread here. However, we'll make sure feedback
    # stays visible during the task execution.
    
    try:
        print(f"DEBUG: Starting launch_task_automation for '{task_description}'")
        # Import the ScreenPrompter class from task_creation_with_command_following
        from task_creation_with_command_following import ScreenPrompter
        
        # Try to read API key from file
        api_key = None
        try:
            with open("api_key.txt", "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            print(f"ERROR reading API key: {e}")
            
            # Cancel any pending clear operations first
            for after_id in feedback_display.tk.call('after', 'info'):
                try:
                    feedback_display.after_cancel(int(after_id))
                    print(f"DEBUG: Cancelled after task with ID {after_id}")
                except ValueError:
                    pass  # Not a numeric ID
            
            error_msg = "Failed to read API key for task automation"
            print(f"DEBUG: Setting error message: '{error_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))
            feedback_display.after(3000, clear_feedback_display)
            
            # Update task status
            track_task(f"Task: {task_description}", "error", "API Key Error")
            
            return
        
        if not api_key:
            # Cancel any pending clear operations first
            for after_id in feedback_display.tk.call('after', 'info'):
                try:
                    feedback_display.after_cancel(int(after_id))
                    print(f"DEBUG: Cancelled after task with ID {after_id}")
                except ValueError:
                    pass  # Not a numeric ID
            
            error_msg = "No API key found for task automation"
            print(f"DEBUG: Setting error message: '{error_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))
            feedback_display.after(3000, clear_feedback_display)
            
            # Update task status
            track_task(f"Task: {task_description}", "error", "No API Key")
            
            return
        
        # Create an instance of ScreenPrompter and send the request
        print(f"DEBUG: Creating ScreenPrompter instance with API key")
        screen_prompter = ScreenPrompter(api_key)
        
        # Cancel any pending clear operations first
        for after_id in feedback_display.tk.call('after', 'info'):
            try:
                feedback_display.after_cancel(int(after_id))
                print(f"DEBUG: Cancelled after task with ID {after_id}")
            except ValueError:
                pass  # Not a numeric ID
        
        # Update UI to show we're starting task automation - ensure it's done in the main thread
        processing_msg = f"Starting task: {task_description}"
        print(f"DEBUG: Setting feedback to '{processing_msg}'")
        feedback_display.after(0, lambda: update_feedback_display(processing_msg, "processing", auto_clear=False))
        
        # Update task status
        track_task(f"Task: {task_description}", "processing", "Processing")
        
        # Send the request to the model - this will block until the task is complete
        print(f"DEBUG: Calling ScreenPrompter.sendRequest with task: '{task_description}'")
        result = screen_prompter.sendRequest(task_description)
        print(f"DEBUG: ScreenPrompter.sendRequest returned: {result}")
        
        # Cancel any pending clear operations before updating with result
        for after_id in feedback_display.tk.call('after', 'info'):
            try:
                feedback_display.after_cancel(int(after_id))
                print(f"DEBUG: Cancelled after task with ID {after_id}")
            except ValueError:
                pass  # Not a numeric ID
        
        # Check if command execution was successful
        if result is not None and isinstance(result, bool) and result is False:
            # If commands were executed successfully without needing a new screenshot
            completion_msg = "Task automation completed successfully"
            print(f"DEBUG: Setting completion message: '{completion_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))
            
            # Update task status
            track_task(f"Task: {task_description}", "success", "Completed")
            
        elif result is not None and isinstance(result, bool) and result is True:
            # If a new screenshot was needed (which means commands were executed)
            completion_msg = "Task automation completed with new screenshot"
            print(f"DEBUG: Setting completion message: '{completion_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))
            
            # Update task status
            track_task(f"Task: {task_description}", "success", "Completed")
            
        else:
            # Default success message if the return value is not as expected
            completion_msg = "Task automation completed"
            print(f"DEBUG: Setting completion message: '{completion_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))
            
            # Update task status
            track_task(f"Task: {task_description}", "success", "Completed")
        
        # Keep the success message visible for 3 seconds before clearing
        print(f"DEBUG: Scheduling clear_feedback_display after 3000ms for '{completion_msg}'")
        feedback_display.after(3000, clear_feedback_display)
        
        # Schedule task removal after a delay
        feedback_display.after(10000, lambda: remove_task(task_id))
        
    except Exception as e:
        print(f"ERROR in run_task_in_thread: {e}")
        
        # Cancel any pending clear operations
        for after_id in feedback_display.tk.call('after', 'info'):
            try:
                feedback_display.after_cancel(int(after_id))
            except ValueError:
                pass  # Not a numeric ID
        
        error_msg = f"Task automation error: {str(e)}"
        print(f"DEBUG: Setting error message: '{error_msg}'")
        feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))
        feedback_display.after(3000, clear_feedback_display)
        
        # Update task status
        track_task(f"Task: {task_description}", "error", "Error")
        
        # Schedule task removal after a delay
        feedback_display.after(10000, lambda: remove_task(task_id))


def update_feedback_display(message, status_type, auto_clear=True):
    """
    Update the feedback display with command execution status
    
    Args:
        message (str): The feedback message to display
        status_type (str): The type of status - 'processing', 'success', or 'error'
        auto_clear (bool): Whether to automatically clear the feedback after a delay (for success messages)
    """
    # Define colors for different status types
    status_colors = {
        "processing": "#3498db",  # Blue
        "success": "#2ecc71",     # Green
        "error": "#e74c3c"        # Red
    }
    
    # Get the color for this status type
    color = status_colors.get(status_type, "#7f8c8d")
    
    # Always use after() to ensure we're updating from the main thread
    def update_display():
        feedback_display.config(
            text=message,
            fg="white",
            bg=color
        )
        
    # Use after(0) to ensure the update happens in the main thread
    feedback_display.after(0, update_display)
    
    # Log the feedback in console too
    print(f"Feedback ({status_type}): {message}")
    
    # Clear the feedback after a delay for success messages if auto_clear is True
    if status_type == "success" and auto_clear:
        feedback_display.after(3000, clear_feedback_display)


def clear_feedback_display():
    """
    Reset the feedback display to the default ready state
    """
    # Use a function to ensure thread safety
    def update_display():
        feedback_display.config(
            text="Ready for next command",
            fg="white",
            bg="#1a2332"
        )
    
    # Use after(0) to ensure the update happens in the main thread
    feedback_display.after(0, update_display)


def save_and_process_audio(audio_data):
    """
    Save and process audio data from the microphone, transcribe it with Whisper,
    and handle recognized voice commands.
    
    Args:
        audio_data (numpy.ndarray): The audio data to process
    """
    # Create a task to track audio processing
    task_id = track_task("Processing audio", "processing", "Transcribing")
    
    try:
        # Ensure the processing message is displayed until we've finished recognition
        print("DEBUG: Setting processing message for audio recognition")
        update_feedback_display("Processing audio...", "processing", auto_clear=False)
        
        processed_audio = preprocess_audio(audio_data)  # Preprocess the audio to enhance recognition quality
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:  # Create a temporary WAV file with '.wav' extension that won't be immediately deleted
            temp_filename = temp_audio_file.name  # Get the name of the temporary file
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio)  # Write the processed audio data to the temporary file

        print("DEBUG: Processing audio with Whisper...")  # Print status message about audio processing
        track_task("Processing audio", "processing", "Recognizing")
        
        segments, _ = model.transcribe(  # Transcribe the audio using the Whisper model and get the segments and info
            temp_filename,  # Path to the audio file
            beam_size=5,  # Beam search size for more accurate transcription
            language="en",  # Force English language for recognition
            condition_on_previous_text=True,  # Use context from previous segments
            no_speech_threshold=0.3  # Threshold for filtering out non-speech
        )
        text = " ".join([segment.text for segment in segments])  # Combine all segments into a single text string
        
        if text.strip():  # Check if any text was recognized
            print(f"DEBUG: Recognized text: {text}")  # Print the recognized text for debugging
            text_input.delete("1.0", tk.END)  # Delete all text in the text input widget
            text_input.insert("1.0", text)  # Insert the recognized text into the text input widget
            
            # Update task status
            track_task("Processing audio", "success", "Text recognized")
            
            update_feedback_display("Speech recognized", "success", auto_clear=True)
            context = get_context_for_speech_command(text)  # Get context to determine if this is likely a false positive

            if context.get("likely_false_positive"):  # Check if the recognition is likely a false positive based on context
                print(f"DEBUG: Ignoring likely false recognition: {text}")  # Log that we're ignoring a likely false recognition
                update_feedback_display("Ignored likely false recognition", "error", auto_clear=True)
                
                # Update task status
                track_task("Processing audio", "error", "False positive")
                
                # Schedule task removal after a delay
                feedback_display.after(5000, lambda: remove_task(task_id))
            else:
                # Process the recognized text as a command - the function will add its own task tracking
                process_voice_command(text)
                
                # Schedule task removal after a delay - we don't need to show both the audio processing
                # and the command processing tasks simultaneously
                feedback_display.after(1000, lambda: remove_task(task_id))
        else:
            print("DEBUG: No speech detected")  # Log that no speech was detected
            text_input.delete("1.0", tk.END)  # Clear the text input widget
            text_input.insert("1.0", "No speech detected")  # Display "No speech detected" message in the text input widget
            update_feedback_display("No speech detected", "error", auto_clear=True)
            
            # Update task status
            track_task("Processing audio", "error", "No speech")
            
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))

        os.unlink(temp_filename)  # Delete the temporary file to clean up

    except Exception as e:
        print(f"ERROR in audio processing: {e}")  # Print error message if any exception occurs during audio processing
        text_input.delete("1.0", tk.END)  # Clear the text input widget
        text_input.insert("1.0", f"Processing error: {str(e)}")  # Display the error message in the text input widget
        update_feedback_display("Audio processing error", "error", auto_clear=True)
        
        # Update task status
        track_task("Processing audio", "error", "Error")
        
        # Schedule task removal after a delay
        feedback_display.after(5000, lambda: remove_task(task_id))


def handle_task_automation(task_description):
    """
    Handle task automation by executing the task_creation_with_command_following.py script in a subprocess
    and updating the UI accordingly.
    
    Args:
        task_description (str): The description of the task to automate
    """
    # Add the task to the tracker
    task_id = track_task(f"Task: {task_description}", "processing", "Starting...")
    
    print(f"DEBUG: handle_task_automation called with task: '{task_description}'")
    
    # Define a nested function to run the task automation in a separate thread
    def run_task_subprocess():
        try:
            # Cancel any pending clear operations to avoid race conditions
            print("DEBUG: Cancelling any pending 'after' calls before setting 'Handing off...' message")
            for after_id in feedback_display.tk.call('after', 'info'):
                try:
                    feedback_display.after_cancel(int(after_id))
                    print(f"DEBUG: Cancelled after task with ID {after_id}")
                except ValueError:
                    pass  # Not a numeric ID
            
            # Update the UI to show we're handing off to task automation
            processing_msg = "Handing off to task automation..."
            print(f"DEBUG: Setting processing message: '{processing_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(processing_msg, "processing", auto_clear=False))
            
            # Update task status
            track_task(f"Task: {task_description}", "processing", "Processing")
            
            # Prepare the subprocess command to run task_creation_with_command_following.py
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_creation_with_command_following.py")
            print(f"DEBUG: Task script path: {script_path}")
            
            # Run the subprocess and capture its output
            command = ["python", script_path, task_description]
            print(f"DEBUG: Running subprocess with command: {command}")
            
            # Using subprocess.run with timeout to prevent hanging
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10-minute timeout
                )
                
                print(f"DEBUG: Subprocess completed with return code: {result.returncode}")
                print(f"DEBUG: Subprocess stdout: {result.stdout}")
                print(f"DEBUG: Subprocess stderr: {result.stderr}")
                
                # Cancel any pending clear operations before updating with the result
                for after_id in feedback_display.tk.call('after', 'info'):
                    try:
                        feedback_display.after_cancel(int(after_id))
                        print(f"DEBUG: Cancelled after task with ID {after_id}")
                    except ValueError:
                        pass  # Not a numeric ID
                
                # Check if the task completed successfully
                if result.returncode == 0:
                    success_msg = "Task automation completed successfully"
                    print(f"DEBUG: Setting success message: '{success_msg}'")
                    feedback_display.after(0, lambda: update_feedback_display(success_msg, "success", auto_clear=False))
                    
                    # Update task status
                    track_task(f"Task: {task_description}", "success", "Completed")
                    
                else:
                    error_msg = f"Task automation failed with code {result.returncode}"
                    print(f"DEBUG: Setting error message: '{error_msg}'")
                    feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))
                    
                    # Update task status
                    track_task(f"Task: {task_description}", "error", f"Failed: {result.returncode}")
                
                # Keep the final message visible for 3 seconds before clearing
                print(f"DEBUG: Scheduling message clear in 3 seconds")
                feedback_display.after(3000, clear_feedback_display)
                
                # Schedule task removal after a delay
                feedback_display.after(10000, lambda: remove_task(task_id))
                
            except subprocess.TimeoutExpired:
                print("DEBUG: Subprocess timed out after 10 minutes")
                
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):
                    try:
                        feedback_display.after_cancel(int(after_id))
                        print(f"DEBUG: Cancelled after task with ID {after_id}")
                    except ValueError:
                        pass  # Not a numeric ID
                
                timeout_msg = "Task automation timed out after 10 minutes"
                print(f"DEBUG: Setting timeout message: '{timeout_msg}'")
                feedback_display.after(0, lambda: update_feedback_display(timeout_msg, "error", auto_clear=False))
                feedback_display.after(3000, clear_feedback_display)
                
                # Update task status
                track_task(f"Task: {task_description}", "error", "Timeout")
                
                # Schedule task removal after a delay
                feedback_display.after(10000, lambda: remove_task(task_id))
                
        except Exception as e:
            print(f"ERROR in run_task_subprocess: {e}")
            
            # Cancel any pending clear operations
            for after_id in feedback_display.tk.call('after', 'info'):
                try:
                    feedback_display.after_cancel(int(after_id))
                    print(f"DEBUG: Cancelled after task with ID {after_id}")
                except ValueError:
                    pass  # Not a numeric ID
            
            error_msg = f"Task automation error: {str(e)}"
            print(f"DEBUG: Setting error message: '{error_msg}'")
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))
            feedback_display.after(3000, clear_feedback_display)
            
            # Update task status
            track_task(f"Task: {task_description}", "error", "Error")
            
            # Schedule task removal after a delay
            feedback_display.after(10000, lambda: remove_task(task_id))
    
    # Start the task in a separate thread to avoid blocking the GUI
    threading.Thread(target=run_task_subprocess, daemon=True).start()


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
            
            # Update feedback display with persistent "Listening for commands..." message
            update_feedback_display("Listening for commands...", "processing", auto_clear=False)
            
            audio_processor = AudioProcessor()  # Create an AudioProcessor instance

            recording_thread = threading.Thread(target=audio_processor.process_audio, daemon=True)  # Create a new thread for audio processing that will run in the background
            recording_thread.start()  # Start the recording thread
            print("Recording thread started")  # Log that recording thread started

        except Exception as e:
            print(f"Error starting recording: {e}")  # Print error message if any exception occurs when starting recording
            status_label.config(text=f"Error: {str(e)}")  # Update status label with the error message
            update_feedback_display(f"Recording error: {str(e)}", "error", auto_clear=True)
            listening_event.clear()  # Clear the listening event to stop audio processing
    else:
        listening_event.clear()  # Stop listening by clearing the event
        status_label.config(text="Press button and speak")  # Update status label to show stopped state
        text_input.delete("1.0", tk.END)  # Clear the text input widget
        text_input.insert("1.0", "Stopped listening")  # Display "Stopped listening" message in the text input widget
        update_feedback_display("Recording stopped", "success", auto_clear=True)
        print("Stopped listening")  # Log that listening stopped


root = tk.Tk()  # Create the main Tkinter window
root.title("CommandFlow")  # Set the window title
root.geometry("850x800")  # Increased height from 600 to 700 to accommodate all elements
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

# Use an absolute path for the microphone icon
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
mic_icon_path = os.path.join(script_dir, "GUI Resources", "mic-icon.png")  # Create absolute path to the icon
mic_image = tk.PhotoImage(file=mic_icon_path)  # Load microphone icon image

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
transcript_frame = tk.Frame(content_area, bg="#1a2332", bd=0, height=280)  # Increased height to use more space
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
feedback_content = tk.Frame(feedback_frame, bg="#1a2332", padx=25, pady=20, height=100)  # Increased height from 80 to 100
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

# After the existing feedback section and before root.mainloop()
# Create a task status section
task_status_frame = tk.Frame(content_area, bg="#1a2332", bd=0)
task_status_frame.pack(fill=tk.X, expand=False, pady=(0, 15))

task_status_header = tk.Frame(task_status_frame, bg="#1a2332", padx=25, pady=15)
task_status_header.pack(fill=tk.X)

task_status_title = tk.Label(task_status_header, text="Active Tasks", 
                            font=("Segoe UI", 14, "bold"), bg="#1a2332", fg="#ffffff")
task_status_title.pack(side=tk.LEFT, anchor=tk.W)

# Add a button to clear completed tasks
clear_tasks_button = tk.Button(task_status_header, text="Clear Completed", 
                              font=("Segoe UI", 10),
                              bg="#2c3445", fg="white",
                              activebackground="#3a4555", activeforeground="white",
                              bd=0, padx=10, pady=2,
                              command=clear_completed_tasks)
clear_tasks_button.pack(side=tk.RIGHT, anchor=tk.E)

separator_task_status = tk.Frame(task_status_frame, height=1, bg="#2c3445")
separator_task_status.pack(fill=tk.X)

# Content area for task status - will contain task items
task_status_content = tk.Frame(task_status_frame, bg="#1a2332", padx=15, pady=10, height=120)  # Increased height from 100 to 120
task_status_content.pack(fill=tk.X)
task_status_content.pack_propagate(False)  # Prevent shrinking

# Initialize the task status display
update_task_status_display()

# Setup a recurring task to clean up old completed tasks
def schedule_task_cleanup():
    clear_completed_tasks()
    root.after(60000, schedule_task_cleanup)  # Run every minute

root.after(60000, schedule_task_cleanup)  # Start the cleanup after 1 minute

root.mainloop()  # Start the Tkinter main event loop
