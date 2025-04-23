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
- 3/30/2025: Added additional commands for creating/executing programs

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
import os   # Import OS for getting file path
import datetime # Import datetime to log date and time for external logging

script_dir = os.path.dirname(__file__)  # Path to the directory the script is in
log_rel_path = "logs\\" + str(datetime.date.today()) + ".txt"   # Relative path to the log file, using current date
log_abs_path = os.path.join(script_dir, log_rel_path)   # Join path to log file to path to current directory
os.makedirs("logs", exist_ok=True)

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

    def make_confirm_action_window(self, nxt_cmd):
        action_confirm_winname = 'Confirm Action'
        cv2.namedWindow(action_confirm_winname)

        FONT, FONT_THICKNESS = cv2.FONT_HERSHEY_COMPLEX, 2
        TEXT_COLOR = (255,255,255) # white
        CMD_COLOR = (200,120,64) # light blue
        
        # Get the texts to put on the frame
        header_txt = "Confirm or cancel action:"
        cmd_txt = f"{nxt_cmd}"
        confirm_txt = "Press `space` to confirm action."
        deny_txt    = "Press `escape` to cancel task execution."

        # Get text sizes
        header_w, header_h     = cv2.getTextSize(header_txt,   FONT, 3, FONT_THICKNESS)[0]
        cmd_w, cmd_h           = cv2.getTextSize(cmd_txt,      FONT, 2, FONT_THICKNESS)[0]
        confirm_w, confirm_h   = cv2.getTextSize(confirm_txt,  FONT, 2, FONT_THICKNESS)[0]
        deny_w, deny_h         = cv2.getTextSize(deny_txt,     FONT, 2, FONT_THICKNESS)[0]
        
        # Make initial grey frame
        frame_w, frame_h = max([header_w, cmd_w, confirm_w, deny_w])+20, np.sum([header_h, cmd_h, confirm_h, deny_h])+400
        frame = np.full((frame_h,frame_w,3), (30,30,30), dtype=np.uint8)

        # Draw the text to the screen
        frame = cv2.putText(frame, header_txt, (frame_w//2 - header_w//2, header_h+20),                         FONT, 3, TEXT_COLOR, FONT_THICKNESS)
        frame = cv2.putText(frame, cmd_txt, (frame_w//2 - cmd_w//2, frame_h//2 + cmd_h//2),                     FONT, 2, CMD_COLOR,  FONT_THICKNESS)
        frame = cv2.putText(frame, confirm_txt, (frame_w//2 - confirm_w//2, frame_h-confirm_h-deny_h-30),       FONT, 2, TEXT_COLOR, FONT_THICKNESS)
        frame = cv2.putText(frame, deny_txt, (frame_w//2 - deny_w//2, frame_h-deny_h-10),                       FONT, 2, TEXT_COLOR, FONT_THICKNESS)

        return frame, action_confirm_winname



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

    def overlayGridOnImg(self, img, center_coord_text=False):  # Method to overlay a grid on the image
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
                text_size_wh = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # Calculate text position
                if center_coord_text: 
                    text_x = x - (text_size_wh[0] // 2) - (cell_w//2)   # for text centerd in cell
                else: 
                    text_x = x - (text_size_wh[0] // 2)               # for text on cell line
                text_y = margin_top - (text_size_wh[1]) # get y position
                cv2.putText(canvas, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA) # put the text on the image

        for i in range(rows + 2):  # Loop through rows to draw horizontal grid lines
            y = margin_top + i * cell_h  # Calculate y position for grid line
            cv2.line(canvas, (margin_left, y), (margin_left + w + cell_w, y), color=grid_color, thickness=1)  # Draw horizontal line

            if i <= rows:  # If within row range
                text = str(i) # Convert row index to string
                text_size_wh = cv2.getTextSize(text, font, font_scale, font_thickness)[0] # Get text size
                
                # Calculate text position
                text_x = margin_left - text_size_wh[0] - 10
                if center_coord_text: 
                    text_y = y + (text_size_wh[1] // 2) - (cell_h//2)   # for text centerd in cell
                else: 
                    text_y = y + (text_size_wh[1] // 2)               # for text on cell line
                cv2.putText(canvas, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        return canvas  # Return the canvas with the grid overlay

    def convImgToB64(self, img):  # Method to convert image to base64
        ret, img_buffer = cv2.imencode('.png', img)  # Encode image to PNG format
        b64_img = base64.b64encode(img_buffer).decode('utf-8')  # Convert to base64 string
        return b64_img  # Return base64 encoded image

    def grid_to_pixel_coordinates(self, row, col):  # Method to convert grid coordinates to pixel coordinates
        pixel_x = self.margin_left + col * self.grid_cell_size_px[0]  # Calculate pixel x-coordinate
        pixel_y = self.margin_top + row * self.grid_cell_size_px[1]  # Calculate pixel y-coordinate
        return pixel_x, pixel_y  # Return pixel coordinates

    def create_and_open_new_python_program(self, command):
        # Create filename
        filename_timestamp = f"{datetime.datetime.now().strftime('%m-%d-%G_%H-%M')}" # get datetime of when program was created
        filename = filename_timestamp + "__" + "generated_script.py"  # create filename by adding the passed in filename to the timestamp string
        
        cmd_flow_dir = os.getcwd() # get absolute working dir
        filepath = rf"{cmd_flow_dir}\output\scripts\{filename}" # get absolute path of the file

        # Need to go through and replace all instances of "\" with "\\" to ensure json readibilty
        esc_filepath = ""
        for ch in filepath:
            if ch == "\\": # if SINGLE backslash
                esc_filepath = esc_filepath + "\\\\" # add DOUBLE backslash to the new string
            else: # if regular char just add it to the new string
                esc_filepath = esc_filepath + ch

        self.generated_script_filepath = esc_filepath     
        
        # Create the commands that need to be executed to create a new script file 
        cmds = f"""
        ```json
        [
            "PRESS_KEY(win+s)",
            "TYPE(Visual Studio Code)",
            "PRESS_KEY(enter)",
            "WAIT(2)",
            "PRESS_KEY(ctrl+n)",
            "PRESS_KEY(ctrl+s)",
            "WAIT(1)",
            "TYPE({self.generated_script_filepath})",
            "PRESS_KEY(enter)",
            "WAIT(2)"
        ]
        ```
        """

        print(f"Cmds to create and open a new script: {cmds}")
        # Execute the commands to create a script
        self.execute_commands(cmds)
        print("Done creating a new script file\n")

    def get_script_from_model(self, command):

        script_message = f"Given the following command delimited by triple backticks, extract the description of the program that the user is requesting, write the python program, and then return only the program as a string: ```{command}```"
        script_prompt = [{ "role": "user", "content": script_message}]

        print("Creating script...")
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
            messages=script_prompt  # Include conversation messages
        )
        response_content = response.choices[0].message.content  # Extract content from response
        print(); print("Script:\n"); print("-"*75)
        print(response_content)  # Print the response content

        # Get only python stuff
        if response_content.startswith("```"):
            script_str = "\n".join(response_content.split("\n")[1:-1]) # trim out the starting ```python and the trailing ```
        return script_str

    def cvt_program_to_cmds(self, program):
        cmds_str = """\
[
"PRESS_KEY(esc)","""

        #HACK-y way of doing this but whatever

        VALID_SYMBOLS = " +-*/=%&|<>:;(),._#"

        # Loop through each character in the program and add the command to type it to the array
        ch_inx = 0
        while ch_inx < len(program):
            # Check for tab
            if program[ch_inx] == "\t":
                cmds_str += '\n"PRESS_KEY(esc)",' # press esc first in case user has autocomplete on
                cmds_str += '\n"PRESS_KEY(tab)",'
            # Check for new line
            elif program[ch_inx] == "\n":
                cmds_str += '\n"PRESS_KEY(esc)",' # press esc first in case user has autocomplete on
                cmds_str += '\n"PRESS_KEY(enter)",'
                cmds_str += '\n"PRESS_KEY(home)",' # to avoid auto-indent messing up the indentation
            # Handle quotes
            elif program[ch_inx] in list("\""):
                cmds_str += f"""\n"PRESS_KEY(\\\")",""" # god it was such a pain to get double quotes to work properly
            elif program[ch_inx] in list("\'"):
                cmds_str += f"""\n"PRESS_KEY(\')","""
            # Else, cur char is a letter, number, or normal symbol 
            elif program[ch_inx].isalnum() or program[ch_inx] in list(VALID_SYMBOLS):
                # Get the full word and then type it all together 
                cur_str = ""
                while program[ch_inx].isalnum() or program[ch_inx] in list(VALID_SYMBOLS):
                    cur_str += program[ch_inx]
                    ch_inx += 1
                    if ch_inx == len(program): break
                                    
                cmds_str += f'\n"TYPE({cur_str})",'
                ch_inx -= 1 # decrement becuase counter will be one higher than it should be for the next char
            
            # Otherwise not a valid char
            else: print(f"Invalid char: {program[ch_inx]}. Ignoring...")
            
            ch_inx += 1 # increment counter on each iteration

        cmds_str += '\n"PRESS_KEY(ctrl+s)"\n]' # save and and add closing bracket

        return cmds_str
                              
    def type_python_program(self, prog):
        
        # Convert the program to command format
        cmds = self.cvt_program_to_cmds(prog)

        # Convert to format that self.execute_commands is expecting
        str_cmds = f"""\
```json
{cmds}
```
"""
        # Execute the commands to create a script
        self.execute_commands(str_cmds)
        print("Done typing python program\n")

    def confirm_action(self, nxt_cmd):
        """
        return True if action was confirmed, False otherwise
        """

        # If next command is None then the CONFIRM_ACTION command was the last command which shouldn't happen but GPT be buggin frfr
        if nxt_cmd is None:
            return True

        print(f"Confirming: {nxt_cmd}")

        # Create pop-up page for user interactivity
        confirm_popup, action_confirm_winname = self.make_confirm_action_window(nxt_cmd)

        # Loop to handle user actions
        running = True
        while running:
            cv2.imshow(action_confirm_winname, confirm_popup) # display window
            key = cv2.waitKey(1) # listen for keystrokes
            # If user presses space, they confirm the action
            if key == ord(' '):
                print("Action confirmed")
                confirmed = True
                running = False
            # If user presses esc, they cancel the task
            elif key == 27: # 27=ascii for escape key
                print("Action canceled")
                confirmed = False
                running = False
        
        # Destory the window
        cv2.destroyWindow(action_confirm_winname)

        return confirmed

    def execute_command(self, command, nxt_cmd=None):  # Method to execute a command

        print(f"command: {command}")

        nxt_cmd_confirmed = True # 

        if command.startswith("ERROR"): # Check for errors
            print("#"*75, "\n"); print(f"ERROR: could not proceed with executing the requested task: {self.prompt}"); print("\n", "#"*75)
        
        elif command.startswith("MOVE_MOUSE"):  # Check if command is to move mouse
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
                print(f"TYPE TEXT: {text}")
                pyautogui.write(text)  # Type the specified text
                print(f"Typed: {text}")  # Print typed text

        elif command.startswith("PRESS_KEY"):  # Check if command is to press a key
            match = re.search(r"PRESS_KEY\((.*)\)", command)  # Match command pattern
            if match:  # If command matches
                key = match.group(1)#.strip('"\'')  # Extract key to press
                if "+" in key:  # Check if it's a shortcut
                    keys = key.split("+")  # Split keys for shortcut
                    pyautogui.hotkey(*keys)  # Press the shortcut keys
                else:  # If it's a single key
                    pyautogui.press(key)  # Press the specified key
                print(f"Pressed key: {key}")  # Print pressed key

        elif command.startswith("WAIT"):
            match = re.search(r"WAIT\((.*)\)", command)  # Match command pattern
            if match:  # If command matches
                sec = match.group(1).strip('"\'')  # Extract how long to wait
                print(f"Waiting {sec}s")
                time.sleep(int(sec)) # sleep specified number of seconds

        elif command.startswith("SCREENSHOT"):  # Check if command is to take a screenshot
            print("Taking a new screenshot...")  # Notify user that a new screenshot will be taken
            return True, nxt_cmd_confirmed  # Indicate that a new screenshot is needed
        
        elif command.startswith("CREATE_SCRIPT"):

            self.create_and_open_new_python_program(command)

        elif command.startswith("WRITE_SCRIPT"):
            # Get the python program as a string
            prog = self.get_script_from_model(self.prompt)
            self.type_python_program(prog)

        elif command.startswith("EXECUTE_SCRIPT"):
            # Create the commands that need to be executed to create a new script file 
            cmds = f"""
            ```json
            [
                "PRESS_KEY(win+s)",
                "TYPE(Command Prompt)",
                "PRESS_KEY(enter)",
                "WAIT(2)",
                "TYPE(python {self.generated_script_filepath})",
                "PRESS_KEY(enter)"
            ]
            ```
            """

            # Execute the commands to create a script
            self.execute_commands(cmds)
            

        elif command.startswith("CONFIRM_ACTION"):
            
            nxt_cmd_confirmed = self.confirm_action(nxt_cmd) # ask user whether the next action can be run
            self.execute_command('WAIT(1)') 

            # If the userr said not to perform the next action 
            if not nxt_cmd_confirmed:
                print("Action was not confirmed, stopping execution...")
            
            return False, nxt_cmd_confirmed

        else: raise ValueError(f"Invalid command {command}")

        return False, nxt_cmd_confirmed  # Indicate no new screenshot is needed and return whether or not the next action was confirmed

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
            for command_inx in range(len(commands)):  # Loop through each command
                print(f"Executing: {commands[command_inx]}")  # Print command being executed
                nxt_cmd = None if command_inx >= len(commands)-1 else commands[command_inx+1]
                take_new_screenshot, nxt_cmd_confirmed = self.execute_command(commands[command_inx], nxt_cmd)  # Execute the command
                if take_new_screenshot:  # If a new screenshot is needed
                    return True  # Indicate that a new screenshot is needed
                # Stop running if user did not confirm next command to be run
                if not nxt_cmd_confirmed:
                    print(f"Next command was not confirmed by the user. Stopping task execution...")
                    break
                time.sleep(0.5)  # Wait before executing the next command
            return False  # Indicate no new screenshot is needed
        except json.JSONDecodeError:  # Handle JSON decoding errors
            print("Failed to parse commands JSON. Raw commands:")  # Notify user of failure
            print(commands_str)  # Print raw commands string
            return False  # Indicate failure

    def createFewShotPrompts(self, verbose=False):
        """
        Create the prompts for the few-shot examples for various commonly used icons/images that need to be located on screen
        """

        few_shot_prompts = []


        ##### Build the few-shot example prompts
        if verbose: print(); print("#"*50); print("Few Shot Examples:")
        else: print("Getting few-shot examples...")

        # Go through the few-shot example directory 
        few_shot_examples_dir = "few_shot_examples"
        for dir in sorted(os.listdir(few_shot_examples_dir)):
            dir_path = os.path.join(few_shot_examples_dir, dir) # eg "few_shot_examples\back_arrow"
            # Only use the directories
            if os.path.isdir(dir_path):

                # Get txt file for prompt
                prompt_txt_path = os.path.join(dir_path, f"{dir}_prompt.txt") # eg "few_shot_examples\back_arrow\back_arrow_prompt.txt"
                # Check that prompt file exists
                if not os.path.isfile(prompt_txt_path): 
                    raise Exception(f"Missing prompt for few-shot examples {dir_path}. Missing file {prompt_txt_path}")
                
                # Get prompt from file
                with open(prompt_txt_path, 'r') as txt_file:
                    prompt = txt_file.read()
                    if verbose: print(prompt)
                # Create header prompt for the example type
                few_shot_examples_header = {
                    "type": "text",
                    "text": prompt
                }
                few_shot_prompts.append(few_shot_examples_header) # add to list of prompts

                # Get images
                valid_img_extensions = ('.png', '.jpg')
                for img_name in sorted([img_path for img_path in os.listdir(dir_path) if img_path.endswith(valid_img_extensions)]): # only loop through image files
                    # Get the full path to the image
                    img_path = os.path.join(dir_path, img_name) # eg. "few_shot_examples/back_arrow/back_arrow_1.png"
                    if verbose: print(img_path)
                    # Convert to b64 so it can be sent in a prompt
                    b64_img = self.convImgToB64(cv2.imread(img_path))
                    # Create the image example prompt 
                    img_prompt = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    }
                    few_shot_prompts.append(img_prompt) # add to list of prompts

        if verbose: print("#"*50); print()

        return few_shot_prompts

    def createCoordinateExamplePrompts(self, example_img_path):
        examples = []

        #### Create coordinate example prompt

        # Check that example image exists
        has_example = os.path.exists(example_img_path)
        if not has_example:
            print(f"Example image {example_img_path} does not exist. Skipping coordinate example")
            return []
        
        b64_coordinate_example = self.convImgToB64(cv2.imread(example_img_path))  # Encode example image
        grid_coordinate_example_prompt = {
            "type": "text",
            "text": "Here's an example screenshot showing grid coordinates. It has a red dot at position (23.25, 13.75) and a blue X at position (26.80, 1.65). Use this as a reference for understanding how coordinates map to positions on the grid."
        }
        # Create coordinate example prompt to send the example image to the model
        grid_coordinate_example_prompt_img = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_coordinate_example}"}
        }
        examples.append(grid_coordinate_example_prompt)
        examples.append(grid_coordinate_example_prompt_img)

        return examples

    def createPromptMessages(self, resume_conversation=False):
        """
        Create the messages that are passed into the api request
        """
        
        # If resuming a conversation, get the previous messages to keep in the context window
        if resume_conversation:
            messages = self.messages.copy()
        # If not resuming, create a fresh messages list
        else:
            messages = []

        # If not resuming a conversation, we need to add the system prompt
        if not resume_conversation:
            ########## Build system prompt - note: can only send images in the user prompt
            system_prompt = { "role": "system", "content": [] }

            # Create the main system prompt that specifies the model's behavior and role
            main_behavior_system_prompt = {
                "type": "text", 
                "text": """
                You are an assistant that helps users control their computer by generating commands based on the user's request and screenshots of their screen-state.

                You will be provided with:
                1. Examples of commonly used icons to help with your detection of them
                2. An example screenshot of a user's screen with an overlayed grid which has coordinates, an example object that is on the screen, the the coordinates where the object is (might not be provided)
                3. The original screenshot WITHOUT any grid overlay
                3. The same screenshot with a numbered grid overlay

                Use the grid overlay to determine precise coordinates, but refer to the original screenshot for visual clarity.

                IMPORTANT: The whole number coordinates (0, 1, 2, etc.) are positioned directly on the grid lines, not in the center of cells.
                When specifying coordinates, use the grid lines as reference points for whole numbers, and use decimal places for positions between lines.

                Available commands:
                0. ERROR() - Use this (and only this) when something goes wrong and the task can not be completed for security reasons or otherwise
                1. MOVE_MOUSE(row, col) - Move the mouse to the specified grid coordinates
                - Coordinates should be specified with 2 decimal places precision (e.g., 5.25, 10.75)
                - This allows for more precise positioning within grid cells
                - IMPORTANT: Only use mouse movements if keyboard commands fail or are not available
                2. CLICK(type) - Click at the current mouse position. Type can be "left" or "right"
                - IMPORTANT: Only use clicks if keyboard commands fail or are not available
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
                6. CREATE_SCRIPT() - Create a new Python script inside of Visual Studio Code
                - This should be the **FIRST** command executed whenever you are asked to create a script
                - This will open a Visual Studio Code window and create a new Python program inside of it
                7. WRITE_SCRIPT() - Write a Python script based on description of the program given in the user prompt
                - This will automatically write the script that the user has described so you just need to have use this command and the script-writing will be done
                - This will be used after the CREATE_SCRIPT() command when making a new program from scratch
                8. EXECUTE_SCRIPT() - Execute the Python script that was created previously
                - This will open the Windows Command Prompt and type the python command to run the script that was prevoiusly created by the CREATE_SCRIPT command
                - When executing **ANY** script, this is the command that you will use
                9. CONFIRM_ACTION() - Prompts the user to confirm the action that you are about to take. Used as a security check so that unintended actions are not taken. 
                - Use this command before running a command that could potentially not be exactly what the user wants.

                IMPORTANT: Keyboard shortcuts and commands are ALWAYS the preferred method for completing tasks. Only use mouse movements and clicks if keyboard commands fail or are not available. This ensures more reliable and efficient task completion.

                Your response should have two sections:

                1. REASONING:
                - Analyze what you see in the screenshot
                - Identify UI elements relevant to the task
                - Consider different approaches to complete the task (including keyboard shortcuts)
                - Explain why you chose specific coordinates or keyboard shortcuts
                - Describe what each element looks like and where it's located

                2. COMMANDS:
                A JSON-formatted list of commands in the exact order they should be executed. For example here are inputs and steps:

{
  "input": "Open File Explorer",
  "steps": [
    "Press Windows key",
    "Press E"
  ]
}

{
  "input": "Open Task Manager",
  "steps": [
    "Press Ctrl + Shift + Esc"
  ]
}

{
  "input": "Launch Google Chrome",
  "steps": [
    "Press Windows key",
    "Type 'chrome'",
    "Press Enter"
  ]
}

{
  "input": "Open an Excel document named test",
  "steps": [
    "Press Windows key",
    "Type 'test.xlsx'",
    "Press Enter"
  ]
}

{
  "input": "Launch Visual Studio Code",
  "steps": [
    "Press Windows key",
    "Type 'Visual Studio Code'",
    "Press Enter"
  ]
}

{
  "input": "Open Notepad",
  "steps": [
    "Press Windows key",
    "Type 'notepad'",
    "Press Enter"
  ]
}

{
  "input": "Open Command Prompt",
  "steps": [
    "Press Windows key",
    "Type 'cmd'",
    "Press Enter"
  ]
}

{
  "input": "Open PowerShell",
  "steps": [
    "Press Windows key",
    "Type 'powershell'",
    "Press Enter"
  ]
}

{
  "input": "Launch Control Panel",
  "steps": [
    "Press Windows key",
    "Type 'control panel'",
    "Press Enter"
  ]
}

{
  "input": "Open Excel and insert a new worksheet",
  "steps": [
    "Press Windows key",
    "Type 'excel'",
    "Press Enter",
    "Press Shift + F11"
  ]
}

{
  "input": "Open Chrome and go to incognito mode",
  "steps": [
    "Press Windows key",
    "Type 'chrome'",
    "Press Enter",
    "Press Ctrl + Shift + N"
  ]
}

{
  "input": "Launch Word and bold selected text",
  "steps": [
    "Press Windows key",
    "Type 'word'",
    "Press Enter",
    "Press Ctrl + B"
  ]
}

{
  "input": "Open Notepad and save the file",
  "steps": [
    "Press Windows key",
    "Type 'notepad'",
    "Press Enter",
    "Press Ctrl + S"
  ]
}

{
  "input": "Open VS Code and run the command palette",
  "steps": [
    "Press Windows key",
    "Type 'Visual Studio Code'",
    "Press Enter",
    "Press Ctrl + Shift + P"
  ]
}

{
  "input": "Minimize all windows",
  "steps": [
    "Press Windows + M"
  ]
}

{
  "input": "Show the desktop",
  "steps": [
    "Press Windows + D"
  ]
}

{
  "input": "Switch between open apps",
  "steps": [
    "Press Alt + Tab"
  ]
}

{
  "input": "Take a screenshot",
  "steps": [
    "Press Windows + Shift + S"
  ]
}

{
  "input": "Lock my PC",
  "steps": [
    "Press Windows + L"
  ]
}

{
  "input": "Open the Run dialog",
  "steps": [
    "Press Windows + R"
  ]
}

                Be precise with coordinates, using the numbered grid on the screenshot. Row numbers (Y-axis) start from 0 at the top left and go down, and column numbers (X-axis) start from 0 at the top left and go right.

                Always provide the most direct and efficient sequence of commands to complete the task.

                If a task is running commands that could do things the user doesn't want, use the "CONFIRM_ACTION()" command before the commands that could cause issues are run. This will have the user confirm the actions you are about to take to ensure you do not accidentally do anthing you shouldn't. For instance, if the user wants you to execute a program, you would open the terminal, type in the command, run the "CONFIRM_ACTION()" command, and then press enter to execute the program if the user confirmed that it is okay. For this example, your list of commands would be something like ["PRESS_KEY(win+s)", "TYPE(Command Prompt)", "TYPE(<command>)", "CONFIRM_ACTION()", "PRESS_KEY(enter)"].
                
                If a task that a user has requested could potentially irreparably damage their system or otherwise cause some sort of harm, do not proceed. Instead, use the "ERROR()" command.
                
                EXAMPLES:
                
                - Example 1: If the user asks you to create a script implementing the famous "fizz-buzz" programming problem, your list of commands should be similar to the following:
                [
                    "CREATE_SCRIPT()",
                    "WRITE_SCRIPT()",
                    "CONFIRM_ACTION()",
                    "EXECUTE_SCRIPT()"
                ]

                - Example 2: If the user asks you to open the Spotify app, your list of commands should be similar to the following:
                [
                    "PRESS_KEY(win+s)",
                    "TYPE(Visual Studio Code)",
                    "PRESS_KEY(enter)"
                ]

                - Example 3: If the user asks you to show them the gambling lines today, your list of commands should be similar to the following:
                [
                    "PRESS_KEY(win+s)",
                    "TYPE(Google Chrome)",
                    "PRESS_KEY(enter)",
                    "WAIT(2)",
                    "TYPE(oddstrader.com)",
                    "PRESS_KEY(enter)"
                ]

                - Example 4: If the user asks you to click on an element on their current webpage, using the grid screenshot you determine it is located in the cell at row X and column Y (where X and Y are placeholders for float values), and then your list of commands should be similar to the following:
                [
                    "MOVE_MOUSE(X,Y)",
                    "CLICK(left)"
                ]

                - Example 5: If the user asks you to delete their system files, your list of commands should be the following:
                [
                    "ERROR()"
                ]
                
                Do a good job or you go to jail.
                """
            }
            # Add the main system prompt to the system prompt container
            system_prompt['content'].append(main_behavior_system_prompt)


        ########## Build user prompt
        user_prompt = { "role": "user", "content": [] }

        # Only add few-shot examples on first prompt
        if not resume_conversation:
            # Build the few-shot example prompts
            few_shot_prompts = self.createFewShotPrompts(verbose=True)

            # Get coordinate example
            coord_example_img_path = "imgs/example_screenshot.jpg"
            coord_example_prompts = self.createCoordinateExamplePrompts(coord_example_img_path)


        # Create the main user prompt to define what the model will actually be trying to acheve
        main_user_prompt = {
            "type": "text",
            "text": f"Please provide the commands needed to complete this task: {self.prompt}\n\nI'm providing two images: the original screenshot and the same screenshot with a grid overlay for coordinate reference. First, reason through the different ways to complete this task, identify the relevant UI elements, and explain your approach. Then provide the specific commands."
        }
        # Create the prmopt to send the original image to the model
        original_img_prompt = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{self.b64_original}"}
        }
        # Create the prmopt to send the image with a grid overlay to the model
        grid_img_prompt = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{self.b64_grid}"}
        }

        # Only add examples on first prompt
        if not resume_conversation:
            # Add few-shot examples for common icons
            [ user_prompt['content'].append(few_shot_prompt) for few_shot_prompt in few_shot_prompts ]
            # Add coordinate example(s)
            [ user_prompt['content'].append(coord_example_prompt) for coord_example_prompt in coord_example_prompts ]

            
        # Add user action prompt and images
        user_prompt["content"].append(main_user_prompt)
        user_prompt["content"].append(original_img_prompt)
        user_prompt["content"].append(grid_img_prompt)

        # Add system and user prompts to messages
        if not resume_conversation: # only add system prompt if resuming a conversation
            messages.append(system_prompt)
        messages.append(user_prompt)

        return messages

    def sendRequest(self, prompt, img_path=None, continue_conversation=False):  # Method to send a request to the OpenAI model
        
        self.prompt = prompt

        # Use auto_screenshot instead of waiting for key press
        if img_path is None:    img = self.auto_screenshot()  # Automatically take a screenshot
        else:                   img = cv2.imread(img_path) # Read passed in image if there is one

        # Save the screenshot with fixed name instead of timestamp to improve determinism
        screenshot_path = f"output/latest_screenshot.png"  # Define path for the screenshot
        cv2.imwrite(screenshot_path, img)  # Save the screenshot to the defined path
        print(f"Screenshot saved to: {screenshot_path}")  # Notify user of saved screenshot
        
        grid_img = self.overlayGridOnImg(img, center_coord_text=True)  # Overlay grid on the screenshot

        grid_path = f"output/latest_grid_screenshot.jpg"  # Define path for the grid overlay image
        cv2.imwrite(grid_path, grid_img)  # Save the grid overlay image
        print(f"Grid overlay image saved to: {grid_path}")  # Notify user of saved grid image

        self.b64_original = self.convImgToB64(img)  # Convert original image to base64
        self.b64_grid = self.convImgToB64(grid_img)  # Convert grid overlay image to base64

        # Get the messages to pass to the model
        self.messages = self.createPromptMessages(resume_conversation=continue_conversation)
        

        # Send the request
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
        print(); print("Model Output:\n"); print("-"*75)
        print(response_content)  # Print the response content

        self.messages.append({  # Append assistant response to messages list
            "role": "assistant",  # Set role to assistant
            "content": response_content  # Include response content
        })

        take_new_screenshot = self.execute_commands(response_content)  # Execute commands from response
        if take_new_screenshot:  # If a new screenshot is needed
            result = self.sendRequest(prompt, continue_conversation=True)  # Recursively send request
            return result  # Return the result of the recursive call
        
        # Log data to external file
        with open(log_abs_path, "a") as logfile:
            # Begin section ('{'), write log ID to file
            logfile.write("{\n")

            # Write the current time, user prompt, and AI response, and close section ('}')
            logfile.write("time: " + str(datetime.datetime.now()) + "\n")
            logfile.write("prompt: " + prompt + "\n\n")
            logfile.write("ai response:\n\n" + response_content + "\n}\n\n")

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
        # Define the default prompt if none provided
        IMG_PROMPT = "Open Windows Search"
    
    print(f"Processing task: {IMG_PROMPT}")  # Print the task being processed

    screenPrompter = ScreenPrompter(API_KEY)  # Create an instance of ScreenPrompter
    result = screenPrompter.sendRequest(IMG_PROMPT)  # Send request to the model
    print(f"Command execution result: {result}")  # Print the result of command execution


    '''Manual testing'''
#     cmds = """```json
# [
#     "TYPE(testing)",
#     "CONFIRM_ACTION()",
#     "TYPE(action confirmed)"
# ]
# ```
# """
#     cmds = """```json
# [
#     "CONFIRM_ACTION()",
#     "TYPE(action confirmed)"
# ]
# ```
# """
#     screenPrompter.execute_commands(cmds)

