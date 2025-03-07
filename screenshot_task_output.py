#!/usr/bin/env python3
"""
Code Artifact: screenshot_task_output.py
Description: AI-powered screenshot task automation tool using OpenAI's GPT model

@author: Christopher Gronewold
@created: 2/14/2025
@revised: 2/27/2025

Revision History:
- 2/14/2025: Initial creation of script (Gianni Louisa)
- 2/27/2025: Modified prompt output handling (Christopher Gronewold)

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

# Import required libraries for image processing, API interaction, and system operations
from openai import OpenAI  # OpenAI API client
import base64  # Base64 encoding/decoding
import cv2  # Image processing
import numpy as np  # Numerical operations
import time  # Time-related functions
import os  # Operating system interactions

class ScreenPrompter:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-08-06"):
        """
        Initialize ScreenPrompter with OpenAI configuration

        @param api_key: OpenAI API authentication key
        @param model: Specific GPT model to use for image analysis
        @raises ValueError: If no API key is provided
        """
        # Validate and set OpenAI API key
        if api_key is not None:
            self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client
        else:
            raise ValueError("No api key for OpenAI was provided. Set api_key=<your-api-key> when creating this object")

        self.model = model  # Store selected model name

        # Create output directory for generated images
        os.makedirs("output", exist_ok=True)  # Create output directory if not exists

    def overlayGridOnImg(self, img, grid_cell_size_px=(50, 50)):
        """
        Overlay a numbered grid on an image for precise coordinate reference

        @param img: Input image to overlay grid on
        @param grid_cell_size_px: Size of grid cells in pixels
        @return: Grid-annotated image with row/column labels
        """
        # Define margins for grid labels
        margin_top = 80  # Top margin for column labels
        margin_left = 80  # Left margin for row labels
        h, w = img.shape[:2]  # Get image height and width

        # Calculate grid dimensions
        cell_w, cell_h = grid_cell_size_px  # Unpack cell dimensions
        rows = int(np.ceil(h / cell_h))  # Calculate number of rows
        cols = int(np.ceil(w / cell_w))  # Calculate number of columns

        # Create white canvas with extra space for grid and labels
        canvas_width = margin_left + w + cell_w  # Add extra width
        canvas_height = margin_top + h + cell_h  # Add extra height
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background

        # Place original image on canvas
        canvas[margin_top:margin_top + h, margin_left:margin_left + w] = img

        # Configure font settings for grid labels
        font = cv2.FONT_HERSHEY_SIMPLEX  # Select font
        font_scale = 0.7  # Font size
        font_thickness = 2  # Font line thickness
        font_color = (0, 0, 200)  # Dark blue color
        grid_color = (100, 100, 100)  # Gray grid line color

        # Add axis labels
        cv2.putText(canvas, "X-axis (Columns)", (margin_left + w // 2 - 100, 30),
                    font, 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Add vertical Y-axis label
        y_label = "Y-axis (Rows)"
        for i, char in enumerate(y_label):
            cv2.putText(canvas, char, (20, margin_top + h // 2 - 100 + i * 25),
                        font, 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Draw vertical grid lines and column numbers
        for i in range(cols + 2):
            x = margin_left + i * cell_w
            cv2.line(canvas, (x, margin_top), (x, margin_top + h + cell_h), color=grid_color, thickness=1)

            # Add column numbers
            if i <= cols:
                text = str(i)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x - text_size[0] // 2
                cv2.putText(canvas, text, (text_x, margin_top - 15),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Draw horizontal grid lines and row numbers
        for i in range(rows + 2):
            y = margin_top + i * cell_h
            cv2.line(canvas, (margin_left, y), (margin_left + w + cell_w, y), color=grid_color, thickness=1)

            # Add row numbers
            if i <= rows:
                text = str(i)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_y = y + text_size[1] // 2
                cv2.putText(canvas, text, (margin_left - text_size[0] - 10, text_y),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        return canvas

    def convImgToB64(self, img):
        """
        Convert image to base64 encoded string for API transmission

        @param img: Input image to encode
        @return: Base64 encoded image string
        """
        ret, img_buffer = cv2.imencode('.png', img)  # Encode image to PNG
        b64_img = base64.b64encode(img_buffer).decode('utf-8')  # Convert to base64
        return b64_img

    def sendRequest(self, img_path, prompt):
        """
        Send screenshot and prompt to OpenAI model for task analysis

        @param img_path: Path to screenshot image
        @param prompt: Task description prompt
        """
        # Load original image
        img = cv2.imread(img_path)  # Read image from file

        # Load example grid image for coordinate reference
        example_img = cv2.imread("imgs/example_screenshot.jpg")

        # Create grid overlay image
        grid_img = self.overlayGridOnImg(img)  # Generate grid-annotated image

        # Save grid overlay image with timestamp
        output_path = f"output/grid_screenshot_{int(time.time())}.jpg"
        cv2.imwrite(output_path, grid_img)  # Save grid image
        print(f"Grid overlay image saved to: {output_path}")

        # Convert images to base64
        b64_original = self.convImgToB64(img)  # Encode original image
        b64_grid = self.convImgToB64(grid_img)  # Encode grid image
        b64_example = self.convImgToB64(example_img)  # Encode example image

        # Send request to OpenAI model with maximum deterministic settings
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,  # Minimum temperature for maximum determinism
            top_p=1.0,  # Use full token distribution
            seed=42,  # Fixed seed for reproducibility
            max_completion_tokens=1500,  # Use max_completion_tokens instead of max_tokens
            n=1,  # Single completion
            stream=False,  # Disable response streaming
            frequency_penalty=0.0,  # No frequency penalty
            presence_penalty=0.0,  # No presence penalty
            logit_bias={},  # No logit bias
            response_format={"type": "text"},  # Explicit text format

            messages=[
                {
                    "role": "system",
                    "content": """
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

                    Your response should have two sections:

                    1. REASONING:
                       - Analyze what you see in the screenshot
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
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here's an example screenshot showing grid coordinates. It has a red dot at position (23.25, 13.75) and a blue X at position (26.80, 1.65). Use this as a reference for understanding how coordinates map to positions on the grid."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_example}"}
                        },
                        {
                            "type": "text",
                            "text": f"Please provide the commands needed to complete this task: {prompt}\n\nI'm providing two images: the original screenshot and the same screenshot with a grid overlay for coordinate reference. First, reason through the different ways to complete this task, identify the relevant UI elements, and explain your approach. Then provide the specific commands."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_original}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_grid}"}
                        }
                    ]
                }
            ]
        )

        # Print model's response
        print(response.choices[0].message.content)

# Main execution block
if __name__ == '__main__':
    # Configuration parameters
    IMG_PATH = "imgs/test_screenshot.png"  # Path to test screenshot
    IMG_PROMPT = "Open Spotify"  # Close the current tab then open a gmail tab.
    API_KEY = ""  # OpenAI API key

    # Initialize and run ScreenPrompter
    screenPrompter = ScreenPrompter(API_KEY)  # Create instance with API key
    screenPrompter.sendRequest(IMG_PATH, IMG_PROMPT)  # Send screenshot request
