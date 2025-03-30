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
    for widget in task_status_content.winfo_children():  # Loop through all widgets in the task status content frame
        widget.destroy()  # Destroy each widget to clear the display
        
    # Check if there are any active tasks
    if not active_tasks:  # If there are no active tasks
        no_tasks_label = tk.Label(  # Create a label showing "No active tasks"
            task_status_content,  # Parent widget
            text="No active tasks",  # Text to display
            font=("Segoe UI", 10),  # Font family and size
            bg="#1a2332",  # Background color (dark blue)
            fg="#7f8c8d",  # Text color (gray)
            anchor=tk.W,  # Anchor text to the west (left)
            padx=5,  # Horizontal padding
            pady=3  # Vertical padding
        )
        no_tasks_label.pack(fill=tk.X, padx=5, pady=2)  # Pack the label to fill horizontally with padding
        return  # Exit the function early
        
    # Add a label for each active task
    for task_id, task_info in active_tasks.items():  # Iterate through each task in the active_tasks dictionary
        task_frame = tk.Frame(task_status_content, bg="#1a2332", padx=0, pady=0)  # Create a frame for each task with dark blue background
        task_frame.pack(fill=tk.X, padx=5, pady=2)  # Pack the frame to fill horizontally with padding
        
        # Status color based on status type
        status_colors = {  # Dictionary mapping status types to colors
            "processing": "#3498db",  # Blue
            "success": "#2ecc71",     # Green
            "error": "#e74c3c",       # Red
            "completed": "#f39c12"    # Orange (for completed tasks waiting to be cleared)
        }
        
        # Status indicator (colored dot)
        status_indicator = tk.Label(  # Create a label for the status indicator
            task_frame,  # Parent widget
            text="●",  # Bullet character as indicator
            font=("Segoe UI", 12),  # Font family and size
            bg="#1a2332",  # Background color (dark blue)
            fg=status_colors.get(task_info["status"], "#7f8c8d"),  # Text color based on status, defaulting to gray
            width=2,  # Width of label
            anchor=tk.W  # Anchor text to the west (left)
        )
        status_indicator.pack(side=tk.LEFT)  # Pack the indicator to the left side
        
        # Task description label
        task_label = tk.Label(  # Create a label for the task description
            task_frame,  # Parent widget
            text=task_info["description"][:40] + "..." if len(task_info["description"]) > 40 else task_info["description"],  # Truncate long descriptions
            font=("Segoe UI", 10),  # Font family and size
            bg="#1a2332",  # Background color (dark blue)
            fg="white",  # Text color (white)
            anchor=tk.W  # Anchor text to the west (left)
        )
        task_label.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Pack the label to the left, filling horizontally
        
        # Add tooltip with full task description
        def show_tooltip(event, text=task_info["description"]):  # Define function to show tooltip with full description
            tooltip = tk.Toplevel(root)  # Create a new top-level window
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")  # Position tooltip near cursor
            tooltip_label = tk.Label(tooltip, text=text, justify=tk.LEFT,  # Create label with full description
                                    background="#1a2332", foreground="white",  # Dark blue background, white text
                                    relief=tk.SOLID, borderwidth=1, padx=5, pady=2)  # Add border and padding
            tooltip_label.pack()  # Pack the tooltip label
            
            def hide_tooltip(_):  # Define function to hide tooltip
                tooltip.destroy()  # Destroy the tooltip window
                
            task_label.bind("<Leave>", hide_tooltip)  # Bind mouse leave event to hide tooltip
            
        task_label.bind("<Enter>", show_tooltip)  # Bind mouse enter event to show tooltip
        
        # Status label
        status_text = task_info.get("status_message", task_info["status"].capitalize())  # Get status message or capitalized status
        status_label = tk.Label(  # Create a label for the status text
            task_frame,  # Parent widget
            text=status_text,  # Text to display
            font=("Segoe UI", 10),  # Font family and size
            bg="#1a2332",  # Background color (dark blue)
            fg=status_colors.get(task_info["status"], "#7f8c8d"),  # Text color based on status, defaulting to gray
            anchor=tk.E,  # Anchor text to the east (right)
            width=12  # Width of label
        )
        status_label.pack(side=tk.RIGHT)  # Pack the label to the right side

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
    global task_id_counter  # Access the global counter for generating unique task IDs
    
    # Check if this task already exists (by description)
    existing_task_id = None  # Initialize variable to store an existing task ID if found
    for task_id, task_info in active_tasks.items():  # Loop through all active tasks
        if task_info["description"] == description:  # If we find a task with the same description
            existing_task_id = task_id  # Store its ID
            break  # Exit the loop
    
    if existing_task_id is not None:  # If we found an existing task
        # Update existing task
        active_tasks[existing_task_id]["status"] = status  # Update its status
        if status_message:  # If a status message was provided
            active_tasks[existing_task_id]["status_message"] = status_message  # Update its status message
        task_id = existing_task_id  # Use the existing task ID
    else:  # If no existing task was found
        # Create new task
        task_id = task_id_counter  # Assign the next available task ID
        task_id_counter += 1  # Increment the counter for the next task
        active_tasks[task_id] = {  # Create a new entry in the active_tasks dictionary
            "description": description,  # Store the task description
            "status": status,  # Store the task status
            "start_time": time.time(),  # Record the current time as the start time
            "status_message": status_message or status.capitalize()  # Use the provided status message or capitalize the status
        }
    
    # Update the task status display
    feedback_display.after(0, update_task_status_display)  # Schedule an update of the task display in the main thread
    
    return task_id  # Return the task ID for future reference

# Function to remove a task from the tracker
def remove_task(task_id):
    """
    Remove a task from the task tracker
    
    Args:
        task_id (int): The ID of the task to remove
    """
    if task_id in active_tasks:  # Check if the task ID exists in the active tasks dictionary
        del active_tasks[task_id]  # Delete the task from the dictionary
        feedback_display.after(0, update_task_status_display)  # Schedule an update of the task display in the main thread

# Function to clear completed tasks
def clear_completed_tasks():
    """
    Remove all completed tasks from the task tracker
    """
    completed_task_ids = [task_id for task_id, task_info in active_tasks.items()  # Create a list of task IDs
                         if task_info["status"] in ["success", "completed", "error"]]  # Where the status is success, completed, or error
    
    for task_id in completed_task_ids:  # Loop through all completed task IDs
        remove_task(task_id)  # Remove each completed task
    
    # Also schedule a periodic cleanup for any tasks that are older than 5 minutes
    current_time = time.time()  # Get the current time
    stale_task_ids = [task_id for task_id, task_info in active_tasks.items()  # Create a list of task IDs 
                     if current_time - task_info.get("start_time", current_time) > 300]  # 5 minutes - where the task is older than 5 minutes
    
    for task_id in stale_task_ids:  # Loop through all stale task IDs
        remove_task(task_id)  # Remove each stale task

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
    command_text = command_text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespace
    
    print(f"DEBUG: Processing voice command: '{command_text}'")  # Print debug information about the command being processed
    
    try:  # Begin try-except block to handle any errors
        # If there's no actual command, just return
        if not command_text:  # Check if the command text is empty
            return  # Exit the function if there's no command to process
            
        # Cancel any pending clear operations first to avoid race conditions
        for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
            try:  # Begin nested try-except block
                feedback_display.after_cancel(int(after_id))  # Try to cancel the callback by its ID
                print(f"DEBUG: Cancelled after task with ID {after_id}")  # Print debug information about cancelled callback
            except ValueError:  # Handle the case where the ID is not a numeric value
                pass  # Not a numeric ID, so skip it
                
        # Show a processing message - make sure it won't auto-clear
        msg = "Processing command..."  # Define the processing message
        print(f"DEBUG: Setting processing message: '{msg}'")  # Print debug information about the message
        update_feedback_display(msg, "processing", auto_clear=False)  # Update the feedback display with the processing message
        
        # Check for different command types and execute the corresponding action
        
        # Task automation command - starts with 'task:'
        if command_text.startswith('task:'):  # Check if the command starts with 'task:'
            task_description = command_text[5:].strip()  # Extract the task description by removing 'task:' prefix
            if task_description:  # Check if there's a task description
                print(f"DEBUG: Detected task automation command: '{task_description}'")  # Print debug information about the task
                # Handle task automation in a separate thread to avoid blocking the GUI
                handle_task_automation(task_description)  # Call the function to handle task automation
            else:  # If there's no task description
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                    try:  # Begin nested try-except block
                        feedback_display.after_cancel(int(after_id))  # Try to cancel the callback by its ID
                    except ValueError:  # Handle the case where the ID is not a numeric value
                        pass  # Not a numeric ID, so skip it
                
                error_msg = "No task description provided. Please specify a task after 'task:'"  # Define error message
                print(f"DEBUG: {error_msg}")  # Print debug information about the error
                update_feedback_display(error_msg, "error")  # Update the feedback display with the error message
                
        # Open website command - starts with 'open:'
        elif command_text.startswith('open:'):  # Check if the command starts with 'open:'
            website = command_text[5:].strip()  # Extract the website by removing 'open:' prefix
            if website:  # Check if there's a website
                # Track this command
                task_id = track_task(f"Open website: {website}", "processing")  # Add this task to the tracker
                
                print(f"DEBUG: Opening website: {website}")  # Print debug information about the website
                if not website.startswith(('http://', 'https://')):  # Check if the website URL starts with http:// or https://
                    website = 'https://' + website  # Add https:// prefix if it's missing
                # Open the website in the default browser
                webbrowser.open(website)  # Open the website in the default web browser
                
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                    try:  # Begin nested try-except block
                        feedback_display.after_cancel(int(after_id))  # Try to cancel the callback by its ID
                    except ValueError:  # Handle the case where the ID is not a numeric value
                        pass  # Not a numeric ID, so skip it
                
                success_msg = f"Opening {website}"  # Define success message
                print(f"DEBUG: {success_msg}")  # Print debug information about the success
                update_feedback_display(success_msg, "success")  # Update the feedback display with the success message
                
                # Update task status
                track_task(f"Open website: {website}", "success", "Opened")  # Update the task status to success
                
                # Schedule task removal after a delay
                feedback_display.after(5000, lambda: remove_task(task_id))  # Schedule task removal after 5 seconds
                
            else:  # If there's no website
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                    try:  # Begin nested try-except block
                        feedback_display.after_cancel(int(after_id))  # Try to cancel the callback by its ID
                    except ValueError:  # Handle the case where the ID is not a numeric value
                        pass  # Not a numeric ID, so skip it
                
                error_msg = "No website provided. Please specify a website after 'open:'"  # Define error message
                print(f"DEBUG: {error_msg}")  # Print debug information about the error
                update_feedback_display(error_msg, "error")  # Update the feedback display with the error message
                
        # Move mouse command
        elif "move mouse" in command_text or "move the mouse" in command_text:  # Check if the command contains 'move mouse' or 'move the mouse'
            # Track this command
            task_id = track_task("Move mouse", "processing")  # Add this task to the tracker
            
            if "top right" in command_text:  # Check if the command specifies 'top right'
                status_label.after(0, lambda: status_label.config(text="Moving mouse to top right"))  # Update status label
                screen_width, _ = pyautogui.size()  # Get screen dimensions
                pyautogui.moveTo(screen_width - 1, 0, duration=0.5)  # Move mouse to top right corner
                
                # Update task status
                track_task("Move mouse", "success", "Top Right")  # Update the task status to success
                
                update_feedback_display("Mouse moved to top right", "success")  # Update feedback display with success message
            else:  # If no specific position is specified
                status_label.after(0, lambda: status_label.config(text="Moving mouse to default position"))  # Update status label
                icon_x, icon_y = 200, 200  # Define default position coordinates
                pyautogui.moveTo(icon_x, icon_y, duration=0.5)  # Move mouse to default position
                
                # Update task status
                track_task("Move mouse", "success", "Default Pos")  # Update the task status to success
                
                update_feedback_display("Mouse moved to default position", "success")  # Update feedback display with success message
                
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))  # Schedule task removal after 5 seconds
            
        # Close window command
        elif "exit window" in command_text or "close window" in command_text:  # Check if the command contains 'exit window' or 'close window'
            # Track this command
            task_id = track_task("Close window", "processing")  # Add this task to the tracker
            
            status_label.after(0, lambda: status_label.config(text="Exiting current window"))  # Update status label
            pyautogui.hotkey("alt", "f4")  # Simulate Alt+F4 key combination to close the active window
            
            # Update task status
            track_task("Close window", "success", "Closed")  # Update the task status to success
            
            update_feedback_display("Window closed", "success")  # Update feedback display with success message
            
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))  # Schedule task removal after 5 seconds
            
        # Command not recognized
        else:  # If none of the above command types match
            # If none of the direct commands matched, try task automation
            print(f"DEBUG: No direct command match, attempting task automation for: '{command_text}'")  # Print debug information
            handle_task_automation(command_text)  # Handle the command as a task automation request
            
    except Exception as e:  # Handle any exceptions that occur during command processing
        print(f"Error processing command: {e}")  # Print error information
        # Cancel any pending clear operations
        for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
            try:  # Begin nested try-except block
                feedback_display.after_cancel(int(after_id))  # Try to cancel the callback by its ID
            except ValueError:  # Handle the case where the ID is not a numeric value
                pass  # Not a numeric ID, so skip it
        
        error_msg = f"Error processing command: {str(e)}"  # Define error message with exception details
        print(f"DEBUG: {error_msg}")  # Print debug information about the error
        update_feedback_display(error_msg, "error")  # Update the feedback display with the error message


def launch_task_automation(task_description):
    """
    Launch the task automation with the given task description
    
    Args:
        task_description (str): The description of the task to automate
    """
    # Add the task to the tracker with initial "processing" status and "Starting..." message
    task_id = track_task(f"Task: {task_description}", "processing", "Starting...")
    
    # This function is already being called in a separate thread by executor.submit(),
    # so we don't need to create another thread here. However, we'll make sure feedback
    # stays visible during the task execution.
    
    try:  # Begin try-except block to handle any errors during task automation
        print(f"DEBUG: Starting launch_task_automation for '{task_description}'")  # Print debug information about starting task
        # Import the ScreenPrompter class from task_creation_with_command_following for AI model integration
        from task_creation_with_command_following import ScreenPrompter  # Import is inside function to avoid circular imports
        
        # Try to read API key from file - needed for the AI model service
        api_key = None  # Initialize api_key variable as None
        try:  # Nested try-except block for API key reading
            with open("api_key.txt", "r") as f:  # Open the API key file in read mode
                api_key = f.read().strip()  # Read the API key and remove any whitespace
        except Exception as e:  # Handle any exceptions during API key reading
            print(f"ERROR reading API key: {e}")  # Print error message with details
            
            # Cancel any pending clear operations first to avoid UI message conflicts
            for after_id in feedback_display.tk.call('after', 'info'):  # Loop through all scheduled "after" callbacks
                try:  # Another nested try-except block
                    feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                    print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                except ValueError:  # Handle case where after_id is not a valid integer
                    pass  # Skip invalid IDs without raising an error
            
            # Update the UI with error message about API key
            error_msg = "Failed to read API key for task automation"  # Define the error message
            print(f"DEBUG: Setting error message: '{error_msg}'")  # Log the error message
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))  # Update UI from main thread
            feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
            
            # Update task status to reflect the API key error
            track_task(f"Task: {task_description}", "error", "API Key Error")  # Mark task as failed with specific error
            
            return  # Exit the function early due to API key error
        
        # Check if API key was successfully retrieved
        if not api_key:  # Check if api_key is still None or empty
            # Cancel any pending clear operations first to avoid UI message conflicts
            for after_id in feedback_display.tk.call('after', 'info'):  # Loop through all scheduled "after" callbacks
                try:  # Nested try-except block
                    feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                    print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                except ValueError:  # Handle case where after_id is not a valid integer
                    pass  # Skip invalid IDs without raising an error
            
            # Update the UI with error message about missing API key
            error_msg = "No API key found for task automation"  # Define the error message
            print(f"DEBUG: Setting error message: '{error_msg}'")  # Log the error message
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))  # Update UI from main thread
            feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
            
            # Update task status to reflect the missing API key
            track_task(f"Task: {task_description}", "error", "No API Key")  # Mark task as failed with specific error
            
            return  # Exit the function early due to missing API key
        
        # Create an instance of ScreenPrompter with the API key
        print(f"DEBUG: Creating ScreenPrompter instance with API key")  # Log that we're creating the ScreenPrompter
        screen_prompter = ScreenPrompter(api_key)  # Initialize the ScreenPrompter with the API key
        
        # Cancel any pending clear operations before showing the processing message
        for after_id in feedback_display.tk.call('after', 'info'):  # Loop through all scheduled "after" callbacks
            try:  # Nested try-except block
                feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
            except ValueError:  # Handle case where after_id is not a valid integer
                pass  # Skip invalid IDs without raising an error
        
        # Update UI to show we're starting task automation - ensure it's done in the main thread
        processing_msg = f"Starting task: {task_description}"  # Define the processing message
        print(f"DEBUG: Setting feedback to '{processing_msg}'")  # Log the message update
        feedback_display.after(0, lambda: update_feedback_display(processing_msg, "processing", auto_clear=False))  # Update UI from main thread
        
        # Update task status to show it's being processed
        track_task(f"Task: {task_description}", "processing", "Processing")  # Update task status
        
        # Send the request to the model - this is a blocking call that will run until the task is complete
        print(f"DEBUG: Calling ScreenPrompter.sendRequest with task: '{task_description}'")  # Log the request
        result = screen_prompter.sendRequest(task_description)  # Send the task to the AI and get the result
        print(f"DEBUG: ScreenPrompter.sendRequest returned: {result}")  # Log the result returned from the AI
        
        # Cancel any pending clear operations before updating with result
        for after_id in feedback_display.tk.call('after', 'info'):  # Loop through all scheduled "after" callbacks
            try:  # Nested try-except block
                feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
            except ValueError:  # Handle case where after_id is not a valid integer
                pass  # Skip invalid IDs without raising an error
        
        # Handle different result types and update UI accordingly
        if result is not None and isinstance(result, bool) and result is False:
            # If commands were executed successfully without needing a new screenshot
            # False means task completed without needing a new screenshot
            completion_msg = "Task automation completed successfully"  # Define success message
            print(f"DEBUG: Setting completion message: '{completion_msg}'")  # Log the message update
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))  # Update UI from main thread
            
            # Update task status to success
            track_task(f"Task: {task_description}", "success", "Completed")  # Mark task as successful
            
        elif result is not None and isinstance(result, bool) and result is True:
            # If a new screenshot was needed (which means commands were executed)
            # True means task completed but needed a new screenshot during execution
            completion_msg = "Task automation completed with new screenshot"  # Define success message
            print(f"DEBUG: Setting completion message: '{completion_msg}'")  # Log the message update
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))  # Update UI from main thread
            
            # Update task status to success
            track_task(f"Task: {task_description}", "success", "Completed")  # Mark task as successful
            
        else:
            # Default success message if the return value is not as expected
            # This handles any other non-error result type
            completion_msg = "Task automation completed"  # Define generic success message
            print(f"DEBUG: Setting completion message: '{completion_msg}'")  # Log the message update
            feedback_display.after(0, lambda: update_feedback_display(completion_msg, "success", auto_clear=False))  # Update UI from main thread
            
            # Update task status to success
            track_task(f"Task: {task_description}", "success", "Completed")  # Mark task as successful
        
        # Keep the success message visible for 3 seconds before clearing
        print(f"DEBUG: Scheduling clear_feedback_display after 3000ms for '{completion_msg}'")  # Log the scheduled clear
        feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
        
        # Schedule task removal after a delay (longer than the feedback message)
        feedback_display.after(10000, lambda: remove_task(task_id))  # Remove task from display after 10 seconds
        
    except Exception as e:  # Handle any unhandled exceptions during task execution
        print(f"ERROR in run_task_in_thread: {e}")  # Log the error with details
        
        # Cancel any pending clear operations to avoid UI message conflicts
        for after_id in feedback_display.tk.call('after', 'info'):  # Loop through all scheduled "after" callbacks
            try:  # Nested try-except block
                feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
            except ValueError:  # Handle case where after_id is not a valid integer
                pass  # Skip invalid IDs without raising an error
        
        # Update UI with error message
        error_msg = f"Task automation error: {str(e)}"  # Define error message with exception details
        print(f"DEBUG: Setting error message: '{error_msg}'")  # Log the error message
        feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))  # Update UI from main thread
        feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
        
        # Update task status to reflect the error
        track_task(f"Task: {task_description}", "error", "Error")  # Mark task as failed with generic error
        
        # Schedule task removal after a delay
        feedback_display.after(10000, lambda: remove_task(task_id))  # Remove task from display after 10 seconds


def update_feedback_display(message, status_type, auto_clear=True):
    """
    Update the feedback display with command execution status
    
    Args:
        message (str): The feedback message to display
        status_type (str): The type of status - 'processing', 'success', or 'error'
        auto_clear (bool): Whether to automatically clear the feedback after a delay (for success messages)
    """
    # Define colors for different status types using standard web color hex codes
    status_colors = {
        "processing": "#3498db",  # Blue color for in-progress tasks
        "success": "#2ecc71",     # Green color for successful tasks
        "error": "#e74c3c"        # Red color for failed tasks
    }
    
    # Get the color for this status type, with a fallback to gray if status type not recognized
    color = status_colors.get(status_type, "#7f8c8d")  # Default to gray (#7f8c8d) if status_type not in dictionary
    
    # Define a nested function to update the display
    # This ensures the update happens atomically within the main thread
    def update_display():
        feedback_display.config(  # Configure the feedback_display label widget
            text=message,  # Set the text content to the provided message
            fg="white",  # Set text color to white for good contrast
            bg=color  # Set background color based on status type
        )
        
    # Use after(0) to ensure the update happens in the main thread
    # This prevents thread-related issues when called from background threads
    feedback_display.after(0, update_display)  # Schedule update_display to run in main thread
    
    # Log the feedback in console too for debugging and tracking
    print(f"Feedback ({status_type}): {message}")  # Print the message with its status type
    
    # Clear the feedback after a delay for success messages if auto_clear is True
    # This ensures temporary success messages don't stay on screen too long
    if status_type == "success" and auto_clear:  # Only auto-clear success messages if enabled
        feedback_display.after(3000, clear_feedback_display)  # Clear after 3 seconds (3000ms)


def clear_feedback_display():
    """
    Reset the feedback display to the default ready state
    """
    # Define a nested function to ensure thread safety when updating the display
    def update_display():
        feedback_display.config(  # Configure the feedback_display label widget
            text="Ready for next command",  # Reset text to default ready message
            fg="white",  # Set text color to white
            bg="#1a2332"  # Reset background to default dark blue color
        )
    
    # Use after(0) to ensure the update happens in the main thread
    # This prevents thread-related issues when called from background threads
    feedback_display.after(0, update_display)  # Schedule update_display to run in main thread


def save_and_process_audio(audio_data):
    """
    Save and process audio data from the microphone, transcribe it with Whisper,
    and handle recognized voice commands.
    
    Args:
        audio_data (numpy.ndarray): The audio data to process
    """
    # Create a task to track audio processing in the task list UI
    task_id = track_task("Processing audio", "processing", "Transcribing")  # Add task with initial transcribing status
    
    try:  # Begin try-except block to handle any errors during audio processing
        # Ensure the processing message is displayed until we've finished recognition
        print("DEBUG: Setting processing message for audio recognition")  # Log that we're starting audio processing
        update_feedback_display("Processing audio...", "processing", auto_clear=False)  # Show processing status in UI
        
        processed_audio = preprocess_audio(audio_data)  # Enhance audio quality through preprocessing function
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:  # Create temporary WAV file
            temp_filename = temp_audio_file.name  # Get the path to the temporary file
            wavfile.write(temp_filename, SAMPLE_RATE, processed_audio)  # Write processed audio to temporary file

        print("DEBUG: Processing audio with Whisper...")  # Log that we're starting Whisper transcription
        track_task("Processing audio", "processing", "Recognizing")  # Update task status to recognizing
        
        # Call the Whisper model to transcribe the audio with optimized parameters
        segments, _ = model.transcribe(  # Get segments of transcribed speech
            temp_filename,  # Path to the audio file
            beam_size=5,  # Beam search size for more accurate transcription
            language="en",  # Force English language for recognition
            condition_on_previous_text=True,  # Use context from previous segments
            no_speech_threshold=0.3  # Threshold for filtering out non-speech
        )
        text = " ".join([segment.text for segment in segments])  # Combine all segments into a single text string
        
        if text.strip():  # Check if any text was recognized (not empty after stripping whitespace)
            print(f"DEBUG: Recognized text: {text}")  # Log the recognized text
            text_input.delete("1.0", tk.END)  # Clear the text input field
            text_input.insert("1.0", text)  # Display the recognized text in the input field
            
            # Update task status to show text was successfully recognized
            track_task("Processing audio", "success", "Text recognized")  # Mark task as successful
            
            update_feedback_display("Speech recognized", "success", auto_clear=True)  # Show success in UI
            context = get_context_for_speech_command(text)  # Get context to check if this is a valid command

            if context.get("likely_false_positive"):  # Check if the recognition is likely incorrect based on context
                print(f"DEBUG: Ignoring likely false recognition: {text}")  # Log that we're ignoring a likely false recognition
                update_feedback_display("Ignored likely false recognition", "error", auto_clear=True)  # Show error in UI
                
                # Update task status to show it was a false positive
                track_task("Processing audio", "error", "False positive")  # Mark task as error with specific reason
                
                # Schedule task removal after a delay
                feedback_display.after(5000, lambda: remove_task(task_id))  # Remove task after 5 seconds
            else:
                # Process the recognized text as a command if it seems valid
                process_voice_command(text)  # Call function to process the command
                
                # Schedule task removal after a delay - we don't need to show both the audio processing
                # and the command processing tasks simultaneously
                feedback_display.after(1000, lambda: remove_task(task_id))  # Remove audio task after 1 second
        else:
            print("DEBUG: No speech detected")  # Log that no speech was detected in the audio
            text_input.delete("1.0", tk.END)  # Clear the text input field
            text_input.insert("1.0", "No speech detected")  # Display "No speech detected" message in input field
            update_feedback_display("No speech detected", "error", auto_clear=True)  # Show error in UI
            
            # Update task status to show no speech was detected
            track_task("Processing audio", "error", "No speech")  # Mark task as error with specific reason
            
            # Schedule task removal after a delay
            feedback_display.after(5000, lambda: remove_task(task_id))  # Remove task after 5 seconds

        os.unlink(temp_filename)  # Delete the temporary audio file to clean up disk space

    except Exception as e:  # Handle any exceptions during audio processing
        print(f"ERROR in audio processing: {e}")  # Log the error with details
        text_input.delete("1.0", tk.END)  # Clear the text input field
        text_input.insert("1.0", f"Processing error: {str(e)}")  # Display the error in the input field
        update_feedback_display("Audio processing error", "error", auto_clear=True)  # Show error in UI
        
        # Update task status to reflect the error
        track_task("Processing audio", "error", "Error")  # Mark task as error with generic reason
        
        # Schedule task removal after a delay
        feedback_display.after(5000, lambda: remove_task(task_id))  # Remove task after 5 seconds


def handle_task_automation(task_description):
    """
    Handle task automation by executing the task_creation_with_command_following.py script in a subprocess
    and updating the UI accordingly.
    
    Args:
        task_description (str): The description of the task to automate
    """
    # Add the task to the tracker with initial status
    task_id = track_task(f"Task: {task_description}", "processing", "Starting...")  # Add task with starting status
    
    print(f"DEBUG: handle_task_automation called with task: '{task_description}'")  # Log task automation start
    
    # Define a nested function to run the task automation in a separate thread
    def run_task_subprocess():
        try:  # Begin try-except block to handle errors in subprocess execution
            # Cancel any pending clear operations to avoid race conditions in UI updates
            print("DEBUG: Cancelling any pending 'after' calls before setting 'Handing off...' message")  # Log cancellation
            for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                try:  # Nested try-except block
                    feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                    print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                except ValueError:  # Handle case where after_id is not a valid integer
                    pass  # Skip invalid IDs without raising an error
            
            # Update the UI to show we're handing off to task automation
            processing_msg = "Handing off to task automation..."  # Define the processing message
            print(f"DEBUG: Setting processing message: '{processing_msg}'")  # Log the message update
            feedback_display.after(0, lambda: update_feedback_display(processing_msg, "processing", auto_clear=False))  # Update UI
            
            # Update task status to show it's being processed
            track_task(f"Task: {task_description}", "processing", "Processing")  # Update task status
            
            # Prepare the subprocess command to run task_creation_with_command_following.py
            # Get the absolute path to the script to ensure it can be found regardless of current directory
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_creation_with_command_following.py")
            print(f"DEBUG: Task script path: {script_path}")  # Log the script path
            
            # Set up the command to run the script with the task description as an argument
            command = ["python", script_path, task_description]  # Create command list for subprocess.run
            print(f"DEBUG: Running subprocess with command: {command}")  # Log the command
            
            # Using subprocess.run with timeout to prevent hanging
            try:  # Nested try-except block to handle subprocess execution
                # Run the subprocess, capturing stdout and stderr, with a 10-minute timeout
                result = subprocess.run(
                    command,  # Command to run
                    capture_output=True,  # Capture output for logging
                    text=True,  # Return output as text rather than bytes
                    timeout=600  # 10-minute timeout to prevent indefinite hanging
                )
                
                # Log the subprocess results for debugging
                print(f"DEBUG: Subprocess completed with return code: {result.returncode}")  # Log return code
                print(f"DEBUG: Subprocess stdout: {result.stdout}")  # Log standard output
                print(f"DEBUG: Subprocess stderr: {result.stderr}")  # Log standard error
                
                # Cancel any pending clear operations before updating with the result
                for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                    try:  # Nested try-except block
                        feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                        print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                    except ValueError:  # Handle case where after_id is not a valid integer
                        pass  # Skip invalid IDs without raising an error
                
                # Check if the task completed successfully
                if result.returncode == 0:  # Return code 0 means success
                    success_msg = "Task automation completed successfully"  # Define success message
                    print(f"DEBUG: Setting success message: '{success_msg}'")  # Log the message update
                    feedback_display.after(0, lambda: update_feedback_display(success_msg, "success", auto_clear=False))  # Update UI
                    
                    # Update task status to success
                    track_task(f"Task: {task_description}", "success", "Completed")  # Mark task as successful
                    
                else:  # Non-zero return code means error
                    error_msg = f"Task automation failed with code {result.returncode}"  # Define error message
                    print(f"DEBUG: Setting error message: '{error_msg}'")  # Log the message update
                    feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))  # Update UI
                    
                    # Update task status to reflect the error
                    track_task(f"Task: {task_description}", "error", f"Failed: {result.returncode}")  # Mark task as error
                
                # Keep the final message visible for 3 seconds before clearing
                print(f"DEBUG: Scheduling message clear in 3 seconds")  # Log the scheduled clear
                feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
                
                # Schedule task removal after a delay
                feedback_display.after(10000, lambda: remove_task(task_id))  # Remove task after 10 seconds
                
            except subprocess.TimeoutExpired:  # Handle the case where subprocess exceeds the timeout
                print("DEBUG: Subprocess timed out after 10 minutes")  # Log the timeout
                
                # Cancel any pending clear operations
                for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                    try:  # Nested try-except block
                        feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                        print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                    except ValueError:  # Handle case where after_id is not a valid integer
                        pass  # Skip invalid IDs without raising an error
                
                # Update UI with timeout message
                timeout_msg = "Task automation timed out after 10 minutes"  # Define timeout message
                print(f"DEBUG: Setting timeout message: '{timeout_msg}'")  # Log the message update
                feedback_display.after(0, lambda: update_feedback_display(timeout_msg, "error", auto_clear=False))  # Update UI
                feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
                
                # Update task status to reflect the timeout
                track_task(f"Task: {task_description}", "error", "Timeout")  # Mark task as error with timeout reason
                
                # Schedule task removal after a delay
                feedback_display.after(10000, lambda: remove_task(task_id))  # Remove task after 10 seconds
                
        except Exception as e:  # Handle any other exceptions during subprocess execution
            print(f"ERROR in run_task_subprocess: {e}")  # Log the error with details
            
            # Cancel any pending clear operations
            for after_id in feedback_display.tk.call('after', 'info'):  # Get all scheduled "after" callbacks
                try:  # Nested try-except block
                    feedback_display.after_cancel(int(after_id))  # Cancel the scheduled callback
                    print(f"DEBUG: Cancelled after task with ID {after_id}")  # Log the cancellation
                except ValueError:  # Handle case where after_id is not a valid integer
                    pass  # Skip invalid IDs without raising an error
            
            # Update UI with error message
            error_msg = f"Task automation error: {str(e)}"  # Define error message with exception details
            print(f"DEBUG: Setting error message: '{error_msg}'")  # Log the error message
            feedback_display.after(0, lambda: update_feedback_display(error_msg, "error", auto_clear=False))  # Update UI
            feedback_display.after(3000, clear_feedback_display)  # Schedule clearing the message after 3 seconds
            
            # Update task status to reflect the error
            track_task(f"Task: {task_description}", "error", "Error")  # Mark task as error with generic reason
            
            # Schedule task removal after a delay
            feedback_display.after(10000, lambda: remove_task(task_id))  # Remove task after 10 seconds
    
    # Start the task in a separate thread to avoid blocking the GUI
    # daemon=True means the thread will be terminated when the main program exits
    threading.Thread(target=run_task_subprocess, daemon=True).start()  # Start the thread for processing task


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
        try:  # Begin try-except block for error handling
            listening_event.set()  # Enable listening by setting the event
            
            # Update feedback display with persistent "Listening for commands..." message
            update_feedback_display("Listening for commands...", "processing", auto_clear=False)  # Show listening status
            
            audio_processor = AudioProcessor()  # Create an AudioProcessor instance to handle audio input

            recording_thread = threading.Thread(target=audio_processor.process_audio, daemon=True)  # Create a new thread for audio processing that will run in the background
            recording_thread.start()  # Start the recording thread
            print("Recording thread started")  # Log that recording thread started

        except Exception as e:  # Handle any exceptions during recording startup
            print(f"Error starting recording: {e}")  # Print error message if any exception occurs when starting recording
            status_label.config(text=f"Error: {str(e)}")  # Update status label with the error message
            update_feedback_display(f"Recording error: {str(e)}", "error", auto_clear=True)  # Show error in feedback display
            listening_event.clear()  # Clear the listening event to stop audio processing
    else:  # If listening is already enabled
        listening_event.clear()  # Stop listening by clearing the event
        status_label.config(text="Press button and speak")  # Update status label to show stopped state
        text_input.delete("1.0", tk.END)  # Clear the text input widget
        text_input.insert("1.0", "Stopped listening")  # Display "Stopped listening" message in the text input widget
        update_feedback_display("Recording stopped", "success", auto_clear=True)  # Show success message for stopping
        print("Stopped listening")  # Log that listening stopped


# GUI Setup Section
root = tk.Tk()  # Create the main Tkinter window
root.title("CommandFlow")  # Set the window title
root.geometry("850x800")  # Set window size to 850x800 pixels
root.configure(bg="#212a38")  # Set the background color to dark blue

main_container = tk.Frame(root, bg="#212a38", padx=0, pady=0)  # Create main container frame with no padding
main_container.pack(fill=tk.BOTH, expand=True)  # Pack the main container to fill the window

sidebar = tk.Frame(main_container, width=340, bg="#ffffff", padx=0, pady=0)  # Create sidebar frame with white background and 340px width
sidebar.pack(side=tk.LEFT, fill=tk.Y)  # Pack the sidebar on the left side, filling vertically
sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking to maintain fixed width

sidebar_content = tk.Frame(sidebar, bg="#ffffff", padx=25, pady=30)  # Add padding container inside sidebar for content
sidebar_content.pack(fill=tk.BOTH, expand=True)  # Pack the sidebar content to fill the sidebar

app_title = tk.Label(sidebar_content, text="CommandFlow", font=("Segoe UI", 22, "bold"),   # Create app title label with modern typography
                    bg="#ffffff", fg="#212a38")  # White background, dark blue text
app_title.pack(anchor=tk.W, pady=(0, 40))  # Pack the app title at the top of the sidebar with bottom padding

# Use an absolute path for the microphone icon
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
mic_icon_path = os.path.join(script_dir, "GUI Resources", "mic-icon.png")  # Create absolute path to the icon
mic_image = tk.PhotoImage(file=mic_icon_path)  # Load microphone icon image

record_button = tk.Button(sidebar_content, image=mic_image, text="",   # Create button with the microphone image
                         compound=tk.CENTER, bd=0, bg="#ffffff",  # Center the image, no border, white background
                         activebackground="#ffffff", command=toggle_record,  # White background when active, call toggle_record on click
                         cursor="hand2", highlightthickness=0)  # Use hand cursor on hover, no highlight
record_button.pack(pady=(0, 30))  # Pack the record button with bottom padding

status_label = tk.Label(sidebar_content, text="Press to speak",  # Create status label with instructions
                       font=("Segoe UI", 12), bg="#ffffff", fg="#4a5568")  # White background, gray text
status_label.pack(pady=(0, 20))  # Pack the status label with bottom padding

content_area = tk.Frame(main_container, bg="#212a38", padx=40, pady=40)  # Create main content area with blue background and padding
content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Pack the content area on the right side, filling the space

content_title = tk.Label(content_area, text="Voice Recognition",   # Add title to content area
                        font=("Segoe UI", 18, "bold"), bg="#212a38", fg="#ffffff")  # Bold font, blue background, white text
content_title.pack(anchor=tk.W, pady=(0, 30))  # Pack the content title at the top of the content area with bottom padding

# Create transcript container with fixed height to prevent it from taking too much space
transcript_frame = tk.Frame(content_area, bg="#1a2332", bd=0, height=280)  # Create frame for transcript with dark blue background and fixed height
transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))  # Pack the transcript frame with bottom padding
transcript_frame.pack_propagate(False)  # Prevent the frame from shrinking to fit its contents

transcript_header = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=20)  # Create header frame for transcript with padding
transcript_header.pack(fill=tk.X)  # Pack the transcript header to fill horizontally

transcript_title = tk.Label(transcript_header, text="Transcription",   # Create label for transcript title
                          font=("Segoe UI", 14), bg="#1a2332", fg="#ffffff")  # White text on dark blue background
transcript_title.pack(anchor=tk.W)  # Pack the transcript title on the left side

separator = tk.Frame(transcript_frame, height=1, bg="#2c3445")  # Create a subtle separator line with height of 1px
separator.pack(fill=tk.X)  # Pack the separator to fill horizontally

transcript_content = tk.Frame(transcript_frame, bg="#1a2332", padx=25, pady=25)  # Create frame for transcript content with padding
transcript_content.pack(fill=tk.BOTH, expand=True)  # Pack the transcript content to fill the transcript frame

text_input = tk.Text(transcript_content,   # Create a Text widget for input and display
                   wrap=tk.WORD,  # Wrap text by words rather than characters
                   fg="#b3c0d1",  # Light blue text color
                   bg="#1a2332",  # Dark blue background
                   font=("Segoe UI", 12),  # Modern font with size 12
                   bd=0,  # No border
                   padx=0,  # No horizontal padding
                   pady=0,  # No vertical padding
                   insertbackground="#ffffff",  # White cursor color
                   selectbackground="#3a4555",  # Selection background color (darker blue)
                   selectforeground="#ffffff",  # Selection text color (white)
                   highlightthickness=0)  # No focus highlight
text_input.pack(fill=tk.BOTH, expand=True)  # Pack the text input to fill the transcript content area
text_input.insert("1.0", "Type or speak your command here...")  # Insert placeholder text at the beginning

def on_focus_in(event):  # Define function for focus-in event (when user clicks into the text field)
    if text_input.get("1.0", "end-1c") == "Type or speak your command here...":  # Check if text contains the placeholder
        text_input.delete("1.0", tk.END)  # Clear the placeholder text
        
def on_focus_out(event):  # Define function for focus-out event (when user clicks away from the text field)
    if text_input.get("1.0", "end-1c").strip() == "":  # Check if text is empty or only whitespace
        text_input.insert("1.0", "Type or speak your command here...")  # Insert the placeholder text

text_input.bind("<FocusIn>", on_focus_in)  # Bind focus-in event to the on_focus_in function
text_input.bind("<FocusOut>", on_focus_out)  # Bind focus-out event to the on_focus_out function

def process_typed_command(event):  # Define function to process commands when Enter is pressed
    command = text_input.get("1.0", "end-1c").strip()  # Get the text from the input widget and remove whitespace
    if command and command != "Type or speak your command here...":  # Check if there's a command and it's not the placeholder
        process_voice_command(command)  # Process the command using the same function for spoken commands
        text_input.delete("1.0", tk.END)  # Clear the input after processing
    return "break"  # Return "break" to prevent default Enter behavior (which would add a new line)

text_input.bind("<Return>", process_typed_command)  # Bind Enter key press to the process_typed_command function

snapshot = get_window_snapshot()  # Get a snapshot of all open windows on the system

all_open_windows = snapshot["all_windows"]  # Get the complete list of all open windows from the snapshot

for window in all_open_windows:  # Loop through all windows to print their details
    print(f"Window: {window.get('title')} - Application: {window.get('app_name')}")  # Print window title and app name

active_app = snapshot["active_window"]["app_name"]  # Get the currently active application name
print(f"You're currently using: {active_app}")  # Print the currently active application name for debugging

# Create a feedback section with more visibility
feedback_frame = tk.Frame(content_area, bg="#1a2332", bd=0)  # Create frame for feedback with dark blue background
feedback_frame.pack(fill=tk.X, expand=False, pady=(0, 15))  # Pack the frame to fill horizontally with bottom padding

feedback_header = tk.Frame(feedback_frame, bg="#1a2332", padx=25, pady=15)  # Create header frame for feedback with padding
feedback_header.pack(fill=tk.X)  # Pack the header to fill horizontally

feedback_title = tk.Label(feedback_header, text="Command Status",  # Create label for feedback title
                         font=("Segoe UI", 14, "bold"), bg="#1a2332", fg="#ffffff")  # Bold font, white text on dark blue
feedback_title.pack(anchor=tk.W)  # Pack the title on the left side

separator_feedback = tk.Frame(feedback_frame, height=1, bg="#2c3445")  # Create a subtle separator line with height of 1px
separator_feedback.pack(fill=tk.X)  # Pack the separator to fill horizontally

# Increase the height of the feedback content area
feedback_content = tk.Frame(feedback_frame, bg="#1a2332", padx=25, pady=20, height=100)  # Frame for feedback with fixed height of 100px
feedback_content.pack(fill=tk.X)  # Pack the content to fill horizontally
feedback_content.pack_propagate(False)  # Prevent shrinking to maintain fixed height

# Make the colored background larger with more padding
feedback_display = tk.Label(feedback_content,  # Create label for displaying feedback
                          text="Ready for commands",  # Initial text
                          font=("Segoe UI", 12),  # Font with size 12
                          bg="#1a2332",  # Dark blue background
                          fg="white",  # White text
                          anchor=tk.CENTER,  # Center the text horizontally
                          padx=20,  # Horizontal padding
                          pady=15,  # Vertical padding
                          wraplength=400,  # Wrap text if longer than 400px
                          justify=tk.CENTER)  # Center-justify the wrapped text
feedback_display.pack(fill=tk.BOTH, expand=True)  # Fill the content area in both directions

# Create a task status section
task_status_frame = tk.Frame(content_area, bg="#1a2332", bd=0)  # Create frame for task status with dark blue background
task_status_frame.pack(fill=tk.X, expand=False, pady=(0, 15))  # Pack the frame to fill horizontally with bottom padding

task_status_header = tk.Frame(task_status_frame, bg="#1a2332", padx=25, pady=15)  # Create header frame for task status with padding
task_status_header.pack(fill=tk.X)  # Pack the header to fill horizontally

task_status_title = tk.Label(task_status_header, text="Active Tasks",  # Create label for task status title
                            font=("Segoe UI", 14, "bold"), bg="#1a2332", fg="#ffffff")  # Bold font, white text on dark blue
task_status_title.pack(side=tk.LEFT, anchor=tk.W)  # Pack the title on the left side

# Add a button to clear completed tasks
clear_tasks_button = tk.Button(task_status_header, text="Clear Completed",  # Create button for clearing completed tasks
                              font=("Segoe UI", 10),  # Font with size 10
                              bg="#2c3445", fg="white",  # Dark blue background, white text
                              activebackground="#3a4555", activeforeground="white",  # Slightly lighter blue when clicked
                              bd=0, padx=10, pady=2,  # No border, with horizontal and vertical padding
                              command=clear_completed_tasks)  # Call clear_completed_tasks function when clicked
clear_tasks_button.pack(side=tk.RIGHT, anchor=tk.E)  # Pack the button on the right side

separator_task_status = tk.Frame(task_status_frame, height=1, bg="#2c3445")  # Create a subtle separator line with height of 1px
separator_task_status.pack(fill=tk.X)  # Pack the separator to fill horizontally

# Content area for task status - will contain task items
task_status_content = tk.Frame(task_status_frame, bg="#1a2332", padx=15, pady=10, height=120)  # Frame for task status content with fixed height
task_status_content.pack(fill=tk.X)  # Pack the content to fill horizontally
task_status_content.pack_propagate(False)  # Prevent shrinking to maintain fixed height

# Initialize the task status display
update_task_status_display()  # Call function to initially populate the task status display

# Setup a recurring task to clean up old completed tasks
def schedule_task_cleanup():  # Define function to periodically clean up tasks
    clear_completed_tasks()  # Call the function to clear completed tasks
    root.after(60000, schedule_task_cleanup)  # Schedule this function to run again after 60000ms (1 minute)

root.after(60000, schedule_task_cleanup)  # Schedule the first cleanup after 1 minute

root.mainloop()  # Start the Tkinter main event loop to handle events and display the GUI
