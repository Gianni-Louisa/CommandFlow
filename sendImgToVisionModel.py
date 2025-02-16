'''
A typical response to a request is of the following form:
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!"
                },
                "logprobs": null,
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
'''


from openai import OpenAI
import base64
import cv2
import numpy as np
import time

class ScreenPrompter:
    def __init__(self, api_key:str=None, model:str="gpt-4o-mini"):

        # Check that the API key is valid
        if api_key is not None: self.client = OpenAI(api_key=api_key)
        else: raise ValueError("No api key for OpenAI was provided. Set api_key=<your-api-key> when creating this object")
        
        # Check model has vision support
        valid_models = ["o1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"] # the openai models with vision support
        if model in valid_models: self.model = model # if model is valid, set object value
        else: raise ValueError(f"Invalid model {model}. Pass in one of the following models that has vision support: {valid_models}")
    
    def overlayGridOnImg(self, img, grid_cell_size_px=(50,50)):
        '''
        Method to overlay a grid over an image

        Parameters:
            img - a cv2 image
            grid_cell_size_px - a tuple of the form (width,height) that specifies the shape of the grid cells in pixels
        '''

        grid_img = img.copy()

        h, w = grid_img.shape[:2]
        cell_w, cell_h = grid_cell_size_px
        rows = int(np.ceil(h / cell_h))
        cols = int(np.ceil(w / cell_w))
        # print(f"rows={rows}; cols={cols}")

        # Draw vertical lines
        for x in np.linspace(start=cell_w, stop=w-cell_w, num=cols-1):
            x = int(round(x))
            cv2.line(grid_img, (x, 0), (x, h), color=(0,0,0), thickness=1)

        # Draw horizontal lines
        for y in np.linspace(start=cell_h, stop=h-cell_h, num=rows-1):
            y = int(round(y))
            cv2.line(grid_img, (0, y), (w, y), color=(0,0,0), thickness=1)

        return grid_img
        
    def convImgToB64(self, img):
        '''
        Method to encode the image to base64 so it can be sent to the model

        Parameters:
            img - a cv2 image
        '''
        ret, img_buffer = cv2.imencode('.png', img)
        b64_img = base64.b64encode(img_buffer).decode('utf-8')
        # with open(image_path, "rb") as image_file:
        #     return base64.b64encode(image_file.read()).decode("utf-8")
        return b64_img
    
    def sendRequest(self, img_path, prompt):
        '''
        Method to send an image and prompt to an openai model
        - https://platform.openai.com/docs/guides/vision

        Parameters:
        img_path - the relative path to the image 
        prompt - the prompt to send along with the image
        '''

        img = cv2.imread(img_path)
        grid_img = self.overlayGridOnImg(img)
        cv2.imwrite(f"output/{time.time()}.jpg", grid_img)
        # return
        b64_img = self.convImgToB64(grid_img)

        response = self.client.chat.completions.create(
            model=f"{self.model}", # specify the model to use in the request

            # Create the message to be passed to the model
            messages=[
                {
                    "role": "developer",
                    "content":  """
                                You are an assistant for an accessibility program that allows users with limited mobility to control a computer using their voice. \
                                You will be provided screenshots of the current state of the user's screen that has a grid overlayed on it and your goal will be to provide information about the objects on the screen and their location in the grid. \
                                This information will then be passed to another AI model that will create a command for controlling the keyboard and mouse to interact with these objects. \
                                When specifying the location of each object in the grid, return a list of cells that contain that object within them in the following form: [(object1_cell1_row, object1_cell1_column),(object1_cell2_row, object1_cell2_column),(object1_cell3_row, object1_cell3_column),...] \
                                Start the cell numbering with column 0 on the far left and row 0 on the top. \
                                Only do this for the five most significant objects on the screen.
                                """ # specify the role/task of the model
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}, # send the image to the model
                        },
                    ],
                }
            ],
        )

        print(response.choices[0])
    

if __name__ == '__main__':

    IMG_PATH = "imgs/test_screenshot.png"
    IMG_PROMPT = ""
    API_KEY = "TODO"

    screenPrompter = ScreenPrompter(API_KEY)
    screenPrompter.sendRequest(IMG_PATH, IMG_PROMPT)