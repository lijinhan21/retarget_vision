import re
import numpy as np
import json
import cv2
import base64

def json_parser(txt):
    '''
    parse json from txt
    '''
    # Regular expression to extract content within ```json and ```
    pattern = r'```json\s*(\{.*?\})\s*```'  # Non-greedy matching for JSON object

    json_data = None
    match = re.findall(pattern, txt, re.DOTALL)  # DOTALL to match across lines
    if match:
        json_str = match[-1]
        try:
            # Parse the JSON string into a Python dictionary
            json_data = json.loads(json_str)
            print("Extracted JSON data:", json_data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON found in the text.")
    
    return json_data

def encode_imgs_from_path(vlm, img_paths):
    '''
        Read in image from img_paths, and convert to base64 string.
    '''
    base64_img_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = vlm.preprocess_image(img, is_rgb=False, encoding='jpg')
        base64_img_list.append(img)
    return base64_img_list