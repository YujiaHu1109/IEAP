import spaces
import torch
import numpy as np
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image
import argparse
import os
import json
import base64
import io
import re
from PIL import ImageFilter
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.ndimage import binary_dilation
import cv2
import openai
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


from src.flux.generate import generate, seed_everything

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

import re

pipe = None
model_dict = {}

def init_flux_pipeline():
    global pipe
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")

def get_model(model_path):
    global model_dict
    if model_path not in model_dict:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_dict[model_path] = (model, tokenizer)
    return model_dict[model_path]

def encode_image_to_datauri(path, size=(512, 512)):
    with Image.open(path).convert('RGB') as img:
        img = img.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError))
)
def cot_with_gpt(image_uri, instruction):
    response = openai.ChatCompletion.create(
        model="gpt-4o",   
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'''
                    Now you are an expert in image editing. Based on the given single image, what atomic image editing instructions should be if the user wants to {instruction}? Let's think step by step. 
                    Atomic instructions include 13 categories as follows:
                    - Add: e.g.: add a car on the road
                    - Remove: e.g.: remove the sofa in the image
                    - Color Change: e.g.: change the color of the shoes to blue
                    - Material Change: e.g.: change the material of the sign like stone
                    - Action Change: e.g.: change the action of the boy to raising hands
                    - Expression Change: e.g.: change the expression to smile
                    - Replace: e.g.: replace the coffee with an apple
                    - Background Change: e.g.: change the background into forest
                    - Appearance Change: e.g.: make the cup have a floral pattern
                    - Move: e.g.: move the plane to the left
                    - Resize: e.g.: enlarge the clock
                    - Tone Transfer: e.g.: change the weather to foggy, change the time to spring
                    - Style Change: e.g.: make the style of the image to cartoon
                    Respond *only* with a numbered list.  
                    Each line must begin with the category in square brackets, then the instruction. Please strictly follow the atomic categories.
                    The operation (what) and the target (to what) are crystal clear.  
                    Do not split replace to add and remove.
                    For example:
                    “1. [Add] add a car on the road\n
                    2. [Color Change] change the color of the shoes to blue\n
                    3. [Move] move the lamp to the left\n"
                    Do not include any extra text, explanations, JSON or markdown—just the list.
                    '''},
                    {
                        "type": "image_url",   
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_uri}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    text = response.choices[0].message.content.strip()
    print(text)

    categories, instructions = extract_instructions(text)
    return categories, instructions

def extract_instructions(text):
    categories = []
    instructions = []
    
    pattern = r'^\s*\d+\.\s*\[(.*?)\]\s*(.*?)$'

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        if match:
            category = match.group(1).strip()
            instruction = match.group(2).strip()
            
            if category and instruction:
                categories.append(category)
                instructions.append(instruction)
    
    return categories, instructions

def extract_last_bbox(result):
    pattern = r'\[?<span data-type="inline-math" data-value="XCcoW15cJ10rKVwnLFxzKlxbXHMqKFxkKylccyosXHMqKFxkKylccyosXHMqKFxkKylccyosXHMqKFxkKylccypcXQ=="></span>\]?'
    matches = re.findall(pattern, result)
    
    if not matches:
        simple_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        simple_matches = re.findall(simple_pattern, result)
        if simple_matches:
            x0, y0, x1, y1 = map(int, simple_matches[-1])
            return [x0, y0, x1, y1]
        else:
            print(f"No bounding boxes found, please try again: {result}")
            return None
    
    last_match = matches[-1]
    x0, y0, x1, y1 = map(int, last_match[1:])
    return x0, y0, x1, y1

@spaces.GPU
def infer_with_DiT(task, image, instruction, category):
    init_flux_pipeline()

    if task == 'RoI Inpainting':
        if category == 'Add' or category == 'Replace':
            lora_path = "weights/add.safetensors"
            added = extract_object_with_gpt(instruction)
            instruction_dit = f"add {added} on the black region"
        elif category == 'Remove' or category == 'Action Change':
            lora_path = "weights/remove.safetensors"
            instruction_dit = f"Fill the hole of the image"
       
        condition = Condition("scene", image, position_delta=(0, 0))
    elif task == 'RoI Editing':
        image = Image.open(image).convert('RGB').resize((512, 512))
        condition = Condition("scene", image, position_delta=(0, -32))
        instruction_dit = instruction
        if category == 'Action Change':
            lora_path = "weights/action.safetensors"
        elif category == 'Expression Change':
            lora_path = "weights/expression.safetensors"
        elif category == 'Add':
            lora_path = "weights/addition.safetensors"
        elif category == 'Material Change':
            lora_path = "weights/material.safetensors"
        elif category == 'Color Change':
            lora_path = "weights/color.safetensors"
        elif category == 'Background Change':
            lora_path = "weights/bg.safetensors"
        elif category == 'Appearance Change':
            lora_path = "weights/appearance.safetensors"
        
    elif task == 'RoI Compositioning':
        lora_path = "weights/fusion.safetensors"
        condition = Condition("scene", image, position_delta=(0, 0))
        instruction_dit = "inpaint the black-bordered region so that the object's edges blend smoothly with the background"

    elif task == 'Global Transformation':
        image = Image.open(image).convert('RGB').resize((512, 512))
        instruction_dit = instruction
        lora_path = "weights/overall.safetensors"

        condition = Condition("scene", image, position_delta=(0, -32))
    else:
        raise ValueError(f"Invalid task: '{task}'")
    
    pipe.unload_lora_weights()
    pipe.load_lora_weights(
        "Cicici1109/IEAP",
        weight_name=lora_path,
        adapter_name="scene",
    )
    
    result_img = generate(
        pipe,
        prompt=instruction_dit,
        conditions=[condition],
        config_path = "train/config/scene_512.yaml",
        num_inference_steps=28,
        height=512,
        width=512,
    ).images[0]

    if task == 'RoI Editing' and category == 'Action Change':
        text_roi = extract_object_with_gpt(instruction)
        instruction_loc = f"<image>Please segment {text_roi}."
        img = result_img

        model, tokenizer = get_model("ByteDance/Sa2VA-8B")

        result = model.predict_forward(
            image=img,
            text=instruction_loc,
            tokenizer=tokenizer,
        )

        prediction = result['prediction']

        if '[SEG]' in prediction and 'prediction_masks' in result:
            pred_mask = result['prediction_masks'][0]   
            pred_mask_np = np.squeeze(np.array(pred_mask))
    
            rows = np.any(pred_mask_np, axis=1)
            cols = np.any(pred_mask_np, axis=0)
            if not np.any(rows) or not np.any(cols):
                print("Warning: Mask is empty, cannot compute bounding box")
                return img
        
            y0, y1 = np.where(rows)[0][[0, -1]]
            x0, x1 = np.where(cols)[0][[0, -1]] 

            changed_instance = crop_masked_region(result_img, pred_mask_np)

            return changed_instance, x0, y1, 1

    return result_img

def load_model(model_path):
    return get_model(model_path)

def extract_object_with_gpt(instruction):
    system_prompt = (
        "You are a helpful assistant that extracts the object or target being edited in an image editing instruction. "
        "Only return a concise noun phrase describing the object. "
        "Examples:\n"
        "- Input: 'Remove the dog' → Output: 'the dog'\n"
        "- Input: 'Add a hat on the dog' → Output: 'a hat'\n"
        "- Input: 'Replace the biggest bear with a tiger' → Output: 'the biggest bear'\n"
        "- Input: 'Change the action of the girl to riding' → Output: 'the girl'\n"
        "- Input: 'Move the red car on the lake' → Output: 'the red car'\n"
        "- Input: 'Minify the carrot on the rabbit's hand' → Output: 'the carrot on the rabbit's hand'\n"
        "- Input: 'Swap the location of the dog and the cat' → Output: 'the dog and the cat'\n"
        "Now extract the object for this instruction:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            temperature=0.2,
            max_tokens=20,
        )
        object_phrase = response.choices[0].message['content'].strip().strip('"')
        print(f"Identified object: {object_phrase}")
        return object_phrase
    except Exception as e:
        print(f"GPT extraction failed: {e}")
        return instruction 
    
def extract_region_with_gpt(instruction):
    system_prompt = (
        "You are a helpful assistant that extracts target region being edited in an image editing instruction. "
        "Only return a concise noun phrase describing the target region. "
        "Examples:\n"
        "- Input: 'Add a red hat to the man on the left' → Output: 'the man on the left'\n"
        "- Input: 'Add a cat beside the dog' → Output: 'the dog'\n"
        "Now extract the target region for this instruction:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            temperature=0.2,
            max_tokens=20,
        )
        object_phrase = response.choices[0].message['content'].strip().strip('"')
        return object_phrase
    except Exception as e:
        print(f"GPT extraction failed: {e}")
        return instruction 
    
def get_masked(mask, image):
    if mask.shape[:2] != image.size[::-1]:  
        raise ValueError(f"Mask size {mask.shape[:2]} does not match image size {image.size}")
    
    image_array = np.array(image)
    image_array[mask] = [0, 0, 0] 

    return Image.fromarray(image_array)

def bbox_to_mask(x0, y0, x1, y1, image_shape=(512, 512), fill_value=True):
    height, width = image_shape
    
    mask = np.zeros((height, width), dtype=bool)
    
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(width, int(x1))
    y1 = min(height, int(y1))
    
    if x0 >= x1 or y0 >= y1:
        print("Warning: Invalid bounding box coordinates")
        return mask

    mask[y0:y1, x0:x1] = fill_value
    
    return mask

def combine_bbox(text, x0, y0, x1, y1):
    bbox = [x0, y0, x1, y1]
    return [(text, bbox)]

def crop_masked_region(image, pred_mask_np):
    if not isinstance(image, Image.Image):
        raise ValueError("The input image is not a PIL Image object")
    if not isinstance(pred_mask_np, np.ndarray) or pred_mask_np.dtype != bool:
        raise ValueError("pred_mask_np must be a NumPy array of boolean type")
    if pred_mask_np.shape[:2] != image.size[::-1]:
        raise ValueError(f"Mask size {pred_mask_np.shape[:2]} does not match image size {image.size}")
    
    image_rgba = image.convert("RGBA")
    image_array = np.array(image_rgba)
    
    rows = np.any(pred_mask_np, axis=1)
    cols = np.any(pred_mask_np, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("Warning: Mask is empty, cannot compute bounding box")
        return image_rgba
    
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    cropped_image = image_array[y0:y1+1, x0:x1+1].copy()
    cropped_mask = pred_mask_np[y0:y1+1, x0:x1+1]
    
    alpha_channel = np.ones(cropped_mask.shape, dtype=np.uint8) * 255
    alpha_channel[~cropped_mask] = 0  
    
    cropped_image[:, :, 3] = alpha_channel

    return Image.fromarray(cropped_image, mode='RGBA')

@spaces.GPU
def roi_localization(image, instruction, category):
    model, tokenizer = get_model("ByteDance/Sa2VA-8B")
    if category == 'Add':
        text_roi = extract_region_with_gpt(instruction)
    else:
        text_roi = extract_object_with_gpt(instruction)
    instruction_loc = f"<image>Please segment {text_roi}."
    img = Image.open(image).convert('RGB').resize((512, 512))
    print(f"Processing image: {os.path.basename(image)}, Instruction: {instruction_loc}")

    result = model.predict_forward(
        image=img,
        text=instruction_loc,
        tokenizer=tokenizer,
    )

    prediction = result['prediction']

    if '[SEG]' in prediction and 'prediction_masks' in result:
        pred_mask = result['prediction_masks'][0]   
        pred_mask_np = np.squeeze(np.array(pred_mask))
        if category == 'Add':
            rows = np.any(pred_mask_np, axis=1)
            cols = np.any(pred_mask_np, axis=0)
            if not np.any(rows) or not np.any(cols):
                print("Warning: Mask is empty, cannot compute bounding box")
                return img  
    
            y0, y1 = np.where(rows)[0][[0, -1]]
            x0, x1 = np.where(cols)[0][[0, -1]] 
            
            bbox = combine_bbox(text_roi, x0, y0, x1, y1)
            x0, y0, x1, y1 = layout_add(bbox, instruction)
            mask = bbox_to_mask(x0, y0, x1, y1)
            masked_img = get_masked(mask, img)
        elif category == 'Move' or category == 'Resize':
            dilated_original_mask = binary_dilation(pred_mask_np, iterations=3)  
            masked_img = get_masked(dilated_original_mask, img)
            
            rows = np.any(pred_mask_np, axis=1)
            cols = np.any(pred_mask_np, axis=0)
            if not np.any(rows) or not np.any(cols):
                print("Warning: Mask is empty, cannot compute bounding box")
                return img  
    
            y0, y1 = np.where(rows)[0][[0, -1]]
            x0, x1 = np.where(cols)[0][[0, -1]] 
            
            bbox = combine_bbox(text_roi, x0, y0, x1, y1)
            x0_new, y0_new, x1_new, y1_new, = layout_change(bbox, instruction)
            scale = (y1_new - y0_new) / (y1 - y0)

            changed_instance = crop_masked_region(img, pred_mask_np)

            return masked_img, changed_instance, x0_new, y1_new, scale
        else:
            dilated_original_mask = binary_dilation(pred_mask_np, iterations=3)  
            masked_img = get_masked(dilated_original_mask, img)
                
        return masked_img

    else:
        print("No valid mask found in the prediction.")
        return None

def fusion(background, foreground, x, y, scale):
    background = background.convert("RGBA")
    bg_width, bg_height = background.size

    fg_width, fg_height = foreground.size
    new_size = (int(fg_width * scale), int(fg_height * scale))
    foreground_resized = foreground.resize(new_size, Image.Resampling.LANCZOS)

    left = x
    top = y - new_size[1]

    canvas = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
    canvas.paste(foreground_resized, (left, top), foreground_resized)
    masked_foreground = process_edge(canvas, left, top, new_size)
    result = Image.alpha_composite(background, masked_foreground)
    
    return result

def process_edge(canvas, left, top, size):
    width, height = size
    
    region = canvas.crop((left, top, left + width, top + height))
    alpha = region.getchannel('A')
    
    dilated_alpha = alpha.filter(ImageFilter.MaxFilter(5)) 
    eroded_alpha = alpha.filter(ImageFilter.MinFilter(3)) 
    
    edge_mask = Image.new('L', (width, height), 0)
    edge_pixels = edge_mask.load()
    dilated_pixels = dilated_alpha.load()
    eroded_pixels = eroded_alpha.load()
    
    for y in range(height):
        for x in range(width):
            if dilated_pixels[x, y] > 0 and eroded_pixels[x, y] == 0:
                edge_pixels[x, y] = 255  
    
    black_edge = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    black_edge.putalpha(edge_mask)
    
    canvas.paste(black_edge, (left, top), black_edge)
    
    return canvas

def combine_text_and_bbox(text_roi, x0, y0, x1, y1):
    return [(text_roi, [x0, y0, x1, y1])]

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError))
)
def layout_add(bbox, instruction):
    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'''
                    You are an intelligent bounding box editor. I will provide you with the current bounding boxes and an add editing instruction. 
                    Your task is to determine the new bounding box of the added object. Let's think step by step.
                    The images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. 
                    The bounding boxes should not go beyond the image boundaries. The new box must be large enough to reasonably encompass the added object in a visually appropriate way, allowing for partial overlap with existing objects when it comes to accessories like hat, necklace. etc.
                    Each bounding box should be in the format of (object name,[top-left x coordinate, top-left y coordinate, bottom-right x coordinate, bottom-right y coordinate]).
                    Only return the bounding box of the newly added object. Do not include the existing bounding boxes.
                    Please consider the semantic information of the layout, preserve semantic relations.
                    If needed, you can make reasonable guesses. Please refer to the examples below:
                    Input bounding boxes: [('a green car', [21, 281, 232, 440])]
                    Editing instruction: Add a bird on the green car.
                    Output bounding boxes: [('a bird', [80, 150, 180, 281])]
                    Input bounding boxes: [('stool', [300, 350, 380, 450])]
                    Editing instruction: Add a cat to the left of the stool.
                    Output bounding boxes: [('a cat', [180, 250, 300, 450])]
                     
                    Here are some examples to illustrate appropriate overlapping for better visual effects:
                    Input bounding boxes: [('the white cat', [200, 300, 320, 420])]
                    Editing instruction: Add a hat on the white cat.
                    Output bounding boxes: [('a hat', [200, 150, 320, 330])]
                    Now, the current bounding boxes is {bbox}, the instruction is {instruction}.
                    '''},
                ],
            }
        ],
        max_tokens=1000,
    )

    result = response.choices[0].message.content.strip()

    bbox = extract_last_bbox(result)
    return bbox

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError))
)
def layout_change(bbox, instruction):
    response = openai.ChatCompletion.create(
        model="gpt-4o",   
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'''
                    You are an intelligent bounding box editor. I will provide you with the current bounding boxes and the editing instruction. 
                    Your task is to generate the new bounding boxes after editing. 
                    The images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. 
                    The bounding boxes should not overlap or go beyond the image boundaries. 
                    Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, bottom-right x coordinate, bottom-right y coordinate]). 
                    Do not add new objects or delete any object provided in the bounding boxes. Do not change the size or the shape of any object unless the instruction requires so.
                    Please consider the semantic information of the layout. 
                    When resizing, keep the bottom-left corner fixed by default. When swaping locations, change according to the center point. 
                    If needed, you can make reasonable guesses. Please refer to the examples below:
                     
                    Input bounding boxes: [('a car', [21, 281, 232, 440])]
                    Editing instruction: Move the car to the right.
                    Output bounding boxes: [('a car', [121, 281, 332, 440])]
                     
                    Input bounding boxes: [("bed", [50, 300, 450, 450]), ("pillow", [200, 200, 300, 230])]
                    Editing instruction: Move the pillow to the left side of the bed.
                    Output bounding boxes: [("bed", [50, 300, 450, 450]), ("pillow", [70, 270, 170, 300])]
                     
                    Input bounding boxes: [("dog", [150, 250, 250, 300])]
                    Editing instruction: Enlarge the dog.
                    Output bounding boxes: [("dog", [150, 225, 300, 300])]
                     
                    Input bounding boxes: [("chair", [100, 350, 200, 450]), ("lamp", [300, 200, 360, 300])]
                    Editing instruction: Swap the location of the chair and the lamp.
                    Output bounding boxes: [("chair", [280, 200, 380, 300]), ("lamp", [120, 350, 180, 450])]
                    Now, the current bounding boxes is {bbox}, the instruction is {instruction}. Let's think step by step, and output the edited layout.
                    '''},
                ],
            }
        ],
        max_tokens=1000,
    )
    result = response.choices[0].message.content.strip()

    bbox = extract_last_bbox(result)
    return bbox

