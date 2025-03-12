# Filename: voiceToText.py
# Description: Voice to Text Demo using SpeechRecognition, Tkinter, and PyAutoGUI.
#              This script provides a simple GUI that allows the user to record voice,
#              convert it to text using the Google Speech Recognition API, and update the
#              interface with the recognized text. In addition, it processes specific voice
#              commands to control the computer (e.g., exit a window, move the mouse to a specific position).
#
# Programmer: Gianni Louisa
# Date Created: 2025-02-10
# Last Revised: 2025-02-10
#
# Revision History:
#   - 2025-02-10 by Gianni Louisa: Initial creation with basic voice recording functionality.
#   - 2025-02-10 by Gianni Louisa: Added toggle functionality for background listening and real-time text updates.
#   - 2025-02-10 by Gianni Louisa: Integrated basic computer control commands (exit window and move mouse).
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

import tkinter as tk # Import the Tkinter library for GUI creation
import speech_recognition as sr # Import the SpeechRecognition library for speech-to-text conversion
import pyautogui  # Library to control the mouse and send keyboard commands

# Global variables to control background listening
listening = False # Flag to indicate if background listening is active
stop_listening = None # Variable to store the stop_listening function

def process_voice_command(command): # Function to process voice commands
    """
    Process the recognized voice command and trigger related computer controls.
    
    Commands:
      - "exit window", "close window", or "quit": Exits the current window.
      - "move mouse to top right": Moves the mouse to the top right of the screen and clicks.
    """
    command = command.lower().strip() # Convert the command to lowercase and remove leading/trailing whitespace 
    if "exit window" in command or "close window" in command or "quit" in command: # If the command is to exit a window
        status_label.after(0, lambda: status_label.config(text="Exiting current window")) # Update the status label to indicate exiting the current window
        pyautogui.hotkey("alt", "f4")  # Simulate Alt + F4 to close the active window
    elif "move mouse" in command: # If the command is to move the mouse
        if "top right" in command: # If the command is to move the mouse to the top right of the screen
            status_label.after(0, lambda: status_label.config(text="Moving mouse to top right")) # Update the status label to indicate moving the mouse to the top right
            # Get screen width and move mouse to top right corner
            screen_width, _ = pyautogui.size() # Get the screen width and height
            # Optionally subtract an offset if needed: e.g., -1 to ensure it's visible.
            pyautogui.moveTo(screen_width - 1, 0, duration=0.5) # Move the mouse to the top right corner of the screen
            pyautogui.click() # Click the mouse
        else: # If the command is not to move the mouse to the top right of the screen
            status_label.after(0, lambda: status_label.config(text="Moving mouse to default icon position")) # Update the status label to indicate moving the mouse to a default icon position
            # Move the mouse to a predefined coordinate (change as needed)
            icon_x, icon_y = 200, 200 # Set the default icon position
            pyautogui.moveTo(icon_x, icon_y, duration=0.5) # Move the mouse to the default icon position
            pyautogui.click() # Click the mouse
    # Additional computer control commands can be added here as needed.

def toggle_record(): # Function to toggle recording
    global listening, stop_listening # Global variables to control background listening
    if not listening: # If background listening is not active
        # Start background listening
        status_label.config(text="Listening...") # Update the status label to indicate listening
        record_button.config(text="Stop Recording") # Update the record button text to indicate stopping recording
        
        recognizer = sr.Recognizer() # Create a recognizer object
        mic = sr.Microphone() # Create a microphone object
        
        def callback(recognizer, audio): # Function to handle the callback from the microphone
            try:
                # Recognize speech using Google Speech Recognition
                recognized_text = recognizer.recognize_google(audio) # Recognize speech using Google Speech Recognition 
                # Update the label with recognized text
                status_label.after(0, lambda: status_label.config(text=f"You said: {recognized_text}")) # Update the status label with the recognized text
                
                # Process the recognized voice command to control the computer
                process_voice_command(recognized_text) # Process the recognized voice command to control the computer
                
            except sr.UnknownValueError: # If the speech is unintelligible
                status_label.after(0, lambda: status_label.config(text="Could not understand audio")) # Update the status label to indicate that the speech was unintelligible
            except sr.RequestError as e: # If there is an error with the Google Speech Recognition API request
                status_label.after(0, lambda: status_label.config(text=f"Request error: {e}")) # Update the status label to indicate that there was an error with the Google Speech Recognition API request
        
        # Start listening in the background
        stop_listening = recognizer.listen_in_background(mic, callback) # Start listening in the background
        listening = True # Set the listening flag to True
    else: # If background listening is active
        # Stop background listening
        if stop_listening is not None:
            stop_listening(wait_for_stop=False) # Stop listening in the background
        listening = False # Set the listening flag to False
        record_button.config(text="Record") # Update the record button text to indicate recording
        status_label.config(text="Stopped listening.") # Update the status label to indicate that listening has stopped

# Set up the main Tkinter window
root = tk.Tk() # Create the main window
root.title("Voice to Text Demo") # Set the title of the window
root.geometry("400x200") # Set the size of the window

# Add a label to display instructions or results
status_label = tk.Label(root, text="Press the button and speak.", wraplength=300) # Create a label to display instructions or results
status_label.pack(pady=20) # Pack the label into the window

# Add a button to start/stop recording
record_button = tk.Button(root, text="Record", command=toggle_record) # Create a button to start/stop recording
record_button.pack() # Pack the button into the window

root.mainloop() # Start the Tkinter event loop

