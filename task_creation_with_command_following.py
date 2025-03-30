#!/usr/bin/env python3
"""
Code Artifact: screenshot_task_output.py
Description: AI-powered screenshot task automation tool using OpenAI's GPT model

@author: Christopher Gronewold
@created: 2/14/2025
@revised: 3/16/2025

Revision History:
- 2/14/2025: Initial creation of script (Gianni Louisa)
- 2/27/2025: Modified prompt output handling (Christopher Gronewold)
- 3/16/2025: Added functionality for command text to execute commands (Christopher Gronewold)
- 3/30/2025: Added task automation(Gianni Louisa)

Preconditions:
- Valid OpenAI API key must be provided
- Screenshot image must exist at specified path
- OpenAI Python library must be installed
- cv2 and numpy libraries must be installed

Postconditions:
- Generates grid-overlaid screenshot
- Sends image to OpenAI model for task analysis
- Outputs command sequence for screen interaction

Error Handling:
- Raises ValueError if no API key is provided
- Handles image processing and encoding errors
- Manages OpenAI API request exceptions

Known Limitations:
- Requires internet connection
- Dependent on model's visual interpretation accuracy
"""

from openai import OpenAI  # Import OpenAI library for API access
import base64  # Import base64 for image encoding
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import time  # Import time for sleep functionality
import os  # Import os for file and directory operations
import pyautogui  # Import pyautogui for GUI automation
import keyboard  # Import keyboard for keyboard event handling
import threading  # Import threading for concurrent execution
from PIL import Image  # Import PIL for image handling
import io  # Import io for byte stream handling
import json  # Import json for JSON handling
import re  # Import re for regular expressions
import sys  # Import sys for command line argument handling


class ScreenPrompter:  # Define the ScreenPrompter class
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-08-06"):  # Constructor with API key and model
        if api_key is not None:  # Check if API key is provided
            self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client with API key
        else:  # If no API key is provided
            raise ValueError("No api key for OpenAI was provided. Set api_key=<your-api-key> when creating this object")  # Raise ValueError

        self.model = model  # Set the model to be used
        os.makedirs("output", exist_ok=True)  # Create output directory if it doesn't exist
        self.screen_width, self.screen_height = pyautogui.size()  # Get screen dimensions
        self.messages = []  # Initialize message list for conversation
        self.grid_cell_size_px = (50, 50)  # Set grid cell size in pixels
        self.margin_top = 80  # Set top margin for grid overlay
        self.margin_left = 80  # Set left margin for grid overlay

    def take_screenshot(self):  # Method to take a screenshot
        screenshot = pyautogui.screenshot()  # Capture the screenshot
        img_byte_arr = io.BytesIO()  # Create a byte stream for the image
        screenshot.save(img_byte_arr, format='PNG')  # Save screenshot to byte stream in PNG format
        img_byte_arr.seek(0)  # Reset byte stream position to the beginning
        img_array = np.array(Image.open(img_byte_arr))  # Convert byte stream to NumPy array
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR format for OpenCV
        return img_array  # Return the image array

    def auto_screenshot(self):
        # Automatically take a screenshot without waiting for a key press
        print("Taking screenshot automatically...")
        return self.take_screenshot()

    def wait_for_screenshot_key(self):  # Method to wait for a key press to take a screenshot
        print("Press Alt+Shift+S to take a screenshot...")  # Prompt user for key press
        screenshot_taken = threading.Event()  # Create an event to signal when screenshot is taken
        screenshot_img = None  # Initialize variable to hold the screenshot image

        def on_screenshot_key():  # Inner function to handle key press
            nonlocal screenshot_img  # Use the outer variable
            print("Taking screenshot...")  # Notify user that screenshot is being taken
            screenshot_img = self.take_screenshot()  # Take the screenshot
            screenshot_taken.set()  # Signal that the screenshot has been taken

        keyboard.add_hotkey('alt+shift+s', on_screenshot_key)  # Set up hotkey for screenshot
        screenshot_taken.wait()  # Wait until the screenshot is taken
        keyboard.remove_hotkey('alt+shift+s')  # Remove the hotkey after use

        return screenshot_img  # Return the taken screenshot image

    def overlayGridOnImg(self, img):  # Method to overlay a grid on the image
        margin_top = self.margin_top  # Get top margin
        margin_left = self.margin_left  # Get left margin
        h, w = img.shape[:2]  # Get height and width of the image

        cell_w, cell_h = self.grid_cell_size_px  # Get grid cell dimensions
        rows = int(np.ceil(h / cell_h))  # Calculate number of rows in the grid
        cols = int(np.ceil(w / cell_w))  # Calculate number of columns in the grid

        canvas_width = margin_left + w + cell_w  # Calculate canvas width
        canvas_height = margin_top + h + cell_h  # Calculate canvas height
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Create a white canvas

        canvas[margin_top:margin_top + h, margin_left:margin_left + w] = img  # Place the image on the canvas

        font = cv2.FONT_HERSHEY_SIMPLEX  # Set font for text
        font_scale = 0.7  # Set font scale
        font_thickness = 2  # Set font thickness
        font_color = (0, 0, 200)  # Set font color (red)
        grid_color = (100, 100, 100)  # Set grid line color (gray)

        cv2.putText(canvas, "X-axis (Columns)", (margin_left + w // 2 - 100, 30),  # Add X-axis label
                    font, 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)  # Set text properties

        y_label = "Y-axis (Rows)"  # Define Y-axis label
        for i, char in enumerate(y_label):  # Loop through each character in the Y-axis label
            cv2.putText(canvas, char, (20, margin_top + h // 2 - 100 + i * 25),  # Add Y-axis labels
                        font, 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)  # Set text properties

        for i in range(cols + 2):  # Loop through columns to draw vertical grid lines
            x = margin_left + i * cell_w  # Calculate x position for grid line
            cv2.line(canvas, (x, margin_top), (x, margin_top + h + cell_h), color=grid_color, thickness=1)  # Draw vertical line

            if i <= cols:  # If within column range
                text = str(i)  # Convert column index to string
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]  # Get text size
                text_x = x - text_size[0] // 2  # Center text above grid line
                cv2.putText(canvas, text, (text_x, margin_top - 15),  # Add column index text
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)  # Set text properties

        for i in range(rows + 2):  # Loop through rows to draw horizontal grid lines
            y = margin_top + i * cell_h  # Calculate y position for grid line
            cv2.line(canvas, (margin_left, y), (margin_left + w + cell_w, y), color=grid_color, thickness=1)  # Draw horizontal line

            if i <= rows:  # If within row range
                text = str(i)  # Convert row index to string
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]  # Get text size
                text_y = y + text_size[1] // 2  # Center text to the left of grid line
                cv2.putText(canvas, text, (margin_left - text_size[0] - 10, text_y),  # Add row index text
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)  # Set text properties

        return canvas  # Return the canvas with the grid overlay

    def convImgToB64(self, img):  # Method to convert image to base64
        ret, img_buffer = cv2.imencode('.png', img)  # Encode image to PNG format
        b64_img = base64.b64encode(img_buffer).decode('utf-8')  # Convert to base64 string
        return b64_img  # Return base64 encoded image

    def grid_to_pixel_coordinates(self, row, col):  # Method to convert grid coordinates to pixel coordinates
        pixel_x = self.margin_left + col * self.grid_cell_size_px[0]  # Calculate pixel x-coordinate
        pixel_y = self.margin_top + row * self.grid_cell_size_px[1]  # Calculate pixel y-coordinate
        return pixel_x, pixel_y  # Return pixel coordinates

    def execute_command(self, command):  # Method to execute a command
        if command.startswith("MOVE_MOUSE"):  # Check if command is to move mouse
            match = re.search(r"MOVE_MOUSE\((\d+\.?\d*),\s*(\d+\.?\d*)\)", command)  # Match command pattern
            if match:  # If command matches
                row, col = float(match.group(1)), float(match.group(2))  # Extract row and column
                pixel_x = self.margin_top + row * self.grid_cell_size_px[1]  # Calculate pixel x-coordinate
                pixel_y = self.margin_left + col * self.grid_cell_size_px[0]  # Calculate pixel y-coordinate
                pyautogui.moveTo(pixel_x, pixel_y)  # Move mouse to calculated position
                print(f"Moved mouse to x: {pixel_x}, y: {pixel_y}")  # Print mouse position

        elif command.startswith("CLICK"):  # Check if command is to click
            match = re.search(r"CLICK\((left|right)\)", command)  # Match command pattern
            if match:  # If command matches
                button = match.group(1)  # Extract button type
                pyautogui.click(button=button)  # Click the specified mouse button
                print(f"Clicked {button} mouse button")  # Print click action

        elif command.startswith("TYPE"):  # Check if command is to type
            match = re.search(r"TYPE\((.*)\)", command)  # Match command pattern
            if match:  # If command matches
                text = match.group(1).strip('"\'')  # Extract text to type
                pyautogui.write(text)  # Type the specified text
                print(f"Typed: {text}")  # Print typed text

        elif command.startswith("PRESS_KEY"):  # Check if command is to press a key
            match = re.search(r"PRESS_KEY\((.*)\)", command)  # Match command pattern
            if match:  # If command matches
                key = match.group(1).strip('"\'')  # Extract key to press
                if "+" in key:  # Check if it's a shortcut
                    keys = key.split("+")  # Split keys for shortcut
                    pyautogui.hotkey(*keys)  # Press the shortcut keys
                else:  # If it's a single key
                    pyautogui.press(key)  # Press the specified key
                print(f"Pressed key: {key}")  # Print pressed key

        elif command.startswith("SCREENSHOT"):  # Check if command is to take a screenshot
            print("Taking a new screenshot...")  # Notify user that a new screenshot will be taken
            return True  # Indicate that a new screenshot is needed

        return False  # Indicate no new screenshot is needed

    def execute_commands(self, commands_str):  # Method to execute a series of commands
        try:  # Try to execute commands
            if "COMMANDS:" in commands_str:  # Check if commands section exists
                commands_section = commands_str.split("COMMANDS:")[1].strip()  # Extract commands section
            else:  # If no commands section
                commands_section = commands_str.strip()  # Clean the string

            cleaned_str = commands_section  # Initialize cleaned string
            if cleaned_str.startswith("```json"):  # Check for JSON code block
                cleaned_str = cleaned_str[7:]  # Remove code block formatting
            if cleaned_str.endswith("```"):  # Check for closing code block
                cleaned_str = cleaned_str[:-3]  # Remove closing code block formatting
            cleaned_str = cleaned_str.strip()  # Clean the string

            commands = json.loads(cleaned_str)  # Load commands from JSON
            for command in commands:  # Loop through each command
                print(f"Executing: {command}")  # Print command being executed
                take_new_screenshot = self.execute_command(command)  # Execute the command
                if take_new_screenshot:  # If a new screenshot is needed
                    return True  # Indicate that a new screenshot is needed
                time.sleep(0.5)  # Wait before executing the next command
            return False  # Indicate no new screenshot is needed
        except json.JSONDecodeError:  # Handle JSON decoding errors
            print("Failed to parse commands JSON. Raw commands:")  # Notify user of failure
            print(commands_str)  # Print raw commands string
            return False  # Indicate failure

    def initialize_system_message(self):  # Method to initialize system message
        system_message = {  # Create system message dictionary
            "role": "system",  # Set role to system
            "content": """  # Set content of the system message
            You are an assistant that helps users control their computer by generating commands based on screenshots.

            You will be provided with:
            1. An example screenshot showing grid coordinates
            2. The original screenshot without any overlay
            3. The same screenshot with a numbered grid overlay

            Use the grid overlay to determine precise coordinates, but refer to the original screenshot for visual clarity.

            IMPORTANT: The whole number coordinates (0, 1, 2, etc.) are positioned directly on the grid lines, not in the center of cells.
            When specifying coordinates, use the grid lines as reference points for whole numbers, and use decimal places for positions between lines.

            Available commands:
            1. MOVE_MOUSE(row, col) - Move the mouse to the specified grid coordinates
               - Coordinates should be specified with 2 decimal places precision (e.g., 5.25, 10.75)
               - This allows for more precise positioning within grid cells
               - Row is the Y coordinate (vertical position from top)
               - Column is the X coordinate (horizontal position from left)
            2. CLICK(type) - Click at the current mouse position. Type can be "left" or "right"
            3. TYPE(text) - Type the specified text
            4. PRESS_KEY(key) - Press a specific keyboard key or keyboard shortcut
               - For single keys: "enter", "escape", "tab", "delete", "backspace", "space"
               - For keyboard shortcuts, use "+" between keys: "ctrl+w", "alt+f4", "ctrl+shift+t"
               - For a sequence of key presses, use separate PRESS_KEY commands for each
               - Examples:
                 * PRESS_KEY(ctrl+w)  # Close a browser tab
                 * PRESS_KEY(alt+f4)  # Close an application
                 * PRESS_KEY(ctrl+c)  # Copy
                 * PRESS_KEY(ctrl+v)  # Paste
            5. SCREENSHOT() - Take a new screenshot to see the updated screen state

            IMPORTANT: Keyboard shortcuts are often the most efficient way to complete tasks. Consider using them when appropriate.

            YOU MUST STRUCTURE YOUR RESPONSE WITH TWO CLEARLY LABELED SECTIONS:

            1. REASONING:
               YOU MUST INCLUDE THIS SECTION. In this section, you should:
               - Analyze what you see in the screenshot in detail
               - Identify UI elements relevant to the task
               - Consider different approaches to complete the task (including keyboard shortcuts)
               - Explain why you chose specific coordinates or keyboard shortcuts
               - Describe what each element looks like and where it's located

            2. COMMANDS:
               A JSON-formatted list of commands in the exact order they should be executed. For example:
               [
                   "MOVE_MOUSE(5.25, 10.75)",
                   "CLICK(left)",
                   "TYPE(Hello world)",
                   "PRESS_KEY(enter)",
                   "SCREENSHOT()"
               ]

            Be precise with coordinates, using the numbered grid on the screenshot. Row numbers (Y-axis) start from 0 at the top, and column numbers (X-axis) start from 0 at the left.

            Always provide the most direct and efficient sequence of commands to complete the task.
            """  # End of content
        }
        self.messages.append(system_message)  # Append system message to messages list

    def sendRequest(self, prompt, continue_conversation=False):  # Method to send a request to the OpenAI model
        if not continue_conversation:  # If not continuing a conversation
            self.messages = []  # Reset messages list
            self.initialize_system_message()  # Initialize system message

        # Use auto_screenshot instead of waiting for key press
        img = self.auto_screenshot()  # Automatically take a screenshot

        # Save the screenshot with fixed name instead of timestamp to improve determinism
        screenshot_path = f"output/latest_screenshot.png"  # Define path for the screenshot
        cv2.imwrite(screenshot_path, img)  # Save the screenshot to the defined path
        print(f"Screenshot saved to: {screenshot_path}")  # Notify user of saved screenshot

        # Check if example image exists before trying to load it
        example_img_path = "imgs/example_screenshot.jpg"
        example_img = None
        has_example = os.path.exists(example_img_path)
        if has_example:
            example_img = cv2.imread(example_img_path)  # Load example image for reference
        
        grid_img = self.overlayGridOnImg(img)  # Overlay grid on the screenshot

        grid_path = f"output/latest_grid_screenshot.jpg"  # Define path for the grid overlay image
        cv2.imwrite(grid_path, grid_img)  # Save the grid overlay image
        print(f"Grid overlay image saved to: {grid_path}")  # Notify user of saved grid image

        b64_original = self.convImgToB64(img)  # Convert original image to base64
        b64_grid = self.convImgToB64(grid_img)  # Convert grid overlay image to base64
        
        # Initialize user message content
        user_message_content = []
        
        # Add example image only if it exists
        if has_example and example_img is not None:
            b64_example = self.convImgToB64(example_img)  # Convert example image to base64
            user_message_content.extend([
                {
                    "type": "text",  # Define type as text
                    "text": "Here's an example screenshot showing grid coordinates. It has a red dot at position (23.25, 13.75) and a blue X at position (26.80, 1.65). Use this as a reference for understanding how coordinates map to positions on the grid."  # Provide context for the example
                },
                {
                    "type": "image_url",  # Define type as image URL
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_example}"}  # Embed example image in base64
                }
            ])
        
        # Add task instructions and screenshots
        user_message_content.extend([
            {
                "type": "text",  # Define type as text
                "text": f"Please provide the commands needed to complete this task: {prompt}\n\nI'm providing two images: the original screenshot and the same screenshot with a grid overlay for coordinate reference. Include both reasoning and commands."  # Request commands from the model
            },
            {
                "type": "image_url",  # Define type as image URL
                "image_url": {"url": f"data:image/jpeg;base64,{b64_original}"}  # Embed original screenshot in base64
            },
            {
                "type": "image_url",  # Define type as image URL
                "image_url": {"url": f"data:image/jpeg;base64,{b64_grid}"}  # Embed grid overlay image in base64
            }
        ])
        
        user_message = {  # Create user message dictionary
            "role": "user",  # Set role to user
            "content": user_message_content  # Set content of the user message
        }

        self.messages.append(user_message)  # Append user message to messages list

        response = self.client.chat.completions.create(  # Send request to OpenAI API
            model=self.model,  # Specify model to use
            temperature=0.0,  # Set temperature for response variability
            top_p=1.0,  # Set top_p for response diversity
            seed=42,  # Set seed for reproducibility
            max_completion_tokens=1500,  # Set maximum tokens for response
            n=1,  # Request one response
            stream=False,  # Disable streaming
            frequency_penalty=0.0,  # Set frequency penalty
            presence_penalty=0.0,  # Set presence penalty
            logit_bias={},  # Set logit bias
            response_format={"type": "text"},  # Set response format
            messages=self.messages  # Include conversation messages
        )

        response_content = response.choices[0].message.content  # Extract content from response
        print(response_content)  # Print the response content

        self.messages.append({  # Append assistant response to messages list
            "role": "assistant",  # Set role to assistant
            "content": response_content  # Include response content
        })

        take_new_screenshot = self.execute_commands(response_content)  # Execute commands from response
        if take_new_screenshot:  # If a new screenshot is needed
            result = self.sendRequest(prompt, continue_conversation=True)  # Recursively send request
            return result  # Return the result of the recursive call
        
        # Return the result of command execution
        # False means commands were executed successfully without needing a new screenshot
        return take_new_screenshot

if __name__ == '__main__':  # Main execution block
    with open("api_key.txt", "r") as f:  # Open API key file
        API_KEY = f.read().strip()  # Read and strip API key

    # Check if a command was passed from another process
    if len(sys.argv) > 1:
        # Use the command passed as argument
        IMG_PROMPT = " ".join(sys.argv[1:])
    else:
        # Default prompt if none provided
        IMG_PROMPT = "Open Windows Search"  # Define the image prompt
    
    print(f"Processing task: {IMG_PROMPT}")

    screenPrompter = ScreenPrompter(API_KEY)  # Create an instance of ScreenPrompter
    result = screenPrompter.sendRequest(IMG_PROMPT)  # Send request to the model
    print(f"Command execution result: {result}")  # Print the result of command execution
