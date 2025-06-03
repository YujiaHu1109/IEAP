import os
import argparse
import json
from PIL import Image
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from utils import encode_image_to_datauri, cot_with_gpt, extract_instructions, infer_with_DiT, roi_localization, fusion
from src.flux.generate import generate, seed_everything

def main():
    parser = argparse.ArgumentParser(description="Evaluate single image + instruction using GPT-4o")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("json_path", help="Path to JSON file containing categories and instructions")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed_everything(args.seed)

    openai.api_key = "YOUR_API_KEY"

    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    os.makedirs("results", exist_ok=True)

    
    #######################################################
    ###         Load instructions from JSON             ###
    #######################################################
    try:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
            categories = data.get('categories', [])
            instructions = data.get('instructions', [])
            
            if not categories or not instructions:
                raise ValueError("JSON file must contain 'categories' and 'instructions' arrays.")
            
            if len(categories) != len(instructions):
                raise ValueError("Length of 'categories' and 'instructions' must match.")
                
            print("Loaded instructions from JSON:")
            for i, (cat, instr) in enumerate(zip(categories, instructions)):
                print(f"Step {i+1}: [{cat}] {instr}")
                
    except Exception as e:
        raise ValueError(f"Failed to load JSON file: {str(e)}")

    ###################################################
    ###          Neural Program Interpreter         ###
    ###################################################
    for i in range(len(categories)):
        if i == 0:
            image = args.image_path
        else:
            image = f"results/{i-1}.png"
        category = categories[i]
        instruction = instructions[i]
        
        if category in ('Add', 'Remove', 'Replace', 'Action Change', 'Move', 'Resize'):
            if category in ('Add', 'Remove', 'Replace'):
                if category == 'Add':
                    edited_image = infer_with_DiT('RoI Editing', image, instruction, category)
                else:
                    mask_image = roi_localization(image, instruction, category)
                    edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, category)
            elif category == 'Action Change':
                mask_image = roi_localization(image, instruction, category)
                edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove')
                changed_instance, x0, y1, scale = infer_with_DiT('RoI Editing', image, instruction, category)
                fusion_image = fusion(edited_image, changed_instance, x0, y1, scale)
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)
            elif category in ('Move', 'Resize'):
                mask_image, changed_instance, x0, y1, scale = roi_localization(image, instruction, category)
                edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove')
                fusion_image = fusion(edited_image, changed_instance, x0, y1, scale)
                fusion_image.save("fusion.png")
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)
      
        elif category in ('Appearance Change', 'Background Change', 'Color Change', 'Material Change', 'Expression Change'):
            edited_image = infer_with_DiT('RoI Editing', image, instruction, category)

        elif category in ('Tone Transfer', 'Style Change'):
            edited_image = infer_with_DiT('Global Transformation', image, instruction, category)
        
        else:
            raise ValueError(f"Invalid category: '{category}'")
        
        image = edited_image
        image.save(f"results/{i}.png")
        print(f"Step {i+1} completed: {category} - {instruction}")


if __name__ == "__main__":
    main()