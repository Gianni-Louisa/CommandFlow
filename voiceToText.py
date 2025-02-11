# Filename: voiceToText.py
# Description: Voice to Text Demo using SpeechRecognition and Tkinter.
#              This script provides a simple GUI that allows the user to record voice,
#              convert it to text using the Google Speech Recognition API, and update the
#              interface with the recognized text. Listening can be toggled on and off.
#
# Programmer: Gianni Louisa
# Date Created: 2023-10-14
# Last Revised: 2023-10-15
#
# Revision History:
#   - 2025-2-10 by Gianni Louisa: Initial creation with basic voice recording functionality. Added toggle functionality for background listening and real-time text updates.
#
# Preconditions:
#   - Python 3.13 installed.
#   - Required packages installed: tkinter, SpeechRecognition, PyAudio.
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
#
# Return Values:
#   - This script does not return any values; it updates the GUI interface based on the recognized speech.
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
#
# Invariants:
#   - The variable "listening" accurately reflects whether background listening is active.
#
# Known Faults:
#   - Real-time transcription updates occur only after a phrase has been completed, rather than continuously.

import tkinter as tk # Import the Tkinter library for GUI creation
import speech_recognition as sr # Import the SpeechRecognition library for speech-to-text conversion

# Global variables to control background listening
listening = False # Flag to indicate if background listening is active
stop_listening = None # Variable to store the stop_listening function

def toggle_record(): # Function to toggle background listening
    global listening, stop_listening # Declare global variables
    if not listening: # If background listening is not active
        # Start background listening
        status_label.config(text="Listening...") # Update the status label to indicate listening
        record_button.config(text="Stop Recording") # Update the record button text to indicate stop recording
        
        recognizer = sr.Recognizer() # Create a recognizer object
        mic = sr.Microphone() # Create a microphone object
        
        def callback(recognizer, audio): # Callback function for speech recognition 
            try:
                # Recognize speech using Google Speech Recognition
                recognized_text = recognizer.recognize_google(audio) # Recognize the speech
                # Update the label (schedule update on the main thread)
                status_label.after(0, lambda: status_label.config(text=f"You said: {recognized_text}")) # Update the status label with the recognized text
            except sr.UnknownValueError: # If the speech is unintelligible
                status_label.after(0, lambda: status_label.config(text="Could not understand audio")) # Update the status label with the error message
            except sr.RequestError as e: # If there is an issue with the Google Speech Recognition API request
                status_label.after(0, lambda: status_label.config(text=f"Request error: {e}")) # Update the status label with the request error
        
        # Optional: specify a phrase_time_limit (e.g., 3 seconds) for shorter segments
        stop_listening = recognizer.listen_in_background(mic, callback)  # , phrase_time_limit=3)
        listening = True # Set the listening flag to True
    else: # If background listening is active
        # Stop background listening
        if stop_listening is not None: # If the stop_listening function is not None
            stop_listening(wait_for_stop=False) # Stop the background listening
        listening = False # Set the listening flag to False
        record_button.config(text="Record") # Update the record button text to indicate record
        status_label.config(text="Stopped listening.") # Update the status label to indicate stopped listening

# Set up the main Tkinter window
root = tk.Tk() # Create the main window
root.title("Voice to Text Demo") # Set the window title
root.geometry("400x200") # Set the window size

# Add a label to display instructions or results
status_label = tk.Label(root, text="Press the button and speak.", wraplength=300) # Create a label to display instructions or results
status_label.pack(pady=20) # Pack the label into the window

# Add a button to start/stop recording
record_button = tk.Button(root, text="Record", command=toggle_record) # Create a button to start/stop recording
record_button.pack() # Pack the button into the window

root.mainloop() # Start the Tkinter event loop

