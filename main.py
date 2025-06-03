import os
import argparse
from PIL import Image
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from utils import encode_image_to_datauri, cot_with_gpt, extract_instructions, infer_with_DiT, roi_localization, fusion
from src.flux.generate import generate, seed_everything

def main():
    parser = argparse.ArgumentParser(description="Evaluate single image + instruction using GPT-4o")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("prompt", help="Original instruction")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed_everything(args.seed)

    openai.api_key = "YOUR_API_KEY"

    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    os.makedirs("results", exist_ok=True)

    
    ###########################################
    ###         CoT -> instructions         ###
    ###########################################

    uri = encode_image_to_datauri(args.image_path)
    categories, instructions = cot_with_gpt(uri, args.prompt)
    print(categories)
    print(instructions)

    # categories = ['Move', 'Resize']
    # instructions = ['Move the woman to the right', 'Minify the woman']

    ###########################################
    ###      Neural Program Interpreter     ###
    ###########################################
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
                    ### RoI Localization
                    mask_image = roi_localization(image, instruction, category)
                    # mask_image.save("mask.png")
                    ### RoI Inpainting
                    edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, category)
            elif category == 'Action Change':
                ### RoI Localization
                mask_image = roi_localization(image, instruction, category)
                ### RoI Inpainting
                edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove') # inpainted bg
                ### RoI Editing
                changed_instance, x0, y1, scale = infer_with_DiT('RoI Editing', image, instruction, category) # action change
                fusion_image = fusion(edited_image, changed_instance, x0, y1, scale)
                ### RoI Compositioning
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)
            elif category in ('Move', 'Resize'):
                ### RoI Localization
                mask_image, changed_instance, x0, y1, scale  = roi_localization(image, instruction, category)
                ### RoI Inpainting
                edited_image= infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove') # inpainted bg
                # changed_instance, bottom_left, scale = layout_change(image, instruction) # move/resize
                fusion_image = fusion(edited_image, changed_instance, x0, y1, scale)
                fusion_image.save("fusion.png")
                ### RoI Compositioning
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)
      
        elif category in ('Appearance Change', 'Background Change', 'Color Change', 'Material Change', 'Expression Change'):
            ### RoI Editing
            edited_image = infer_with_DiT('RoI Editing', image, instruction, category)

        elif category in ('Tone Transfer', 'Style Change'):
            ### Global Transformation
            edited_image = infer_with_DiT('Global Transformation', image, instruction, category)
        
        else:
            raise ValueError(f"Invalid category: '{category}'")
        
        image = edited_image
        image.save(f"results/{i}.png")


if __name__ == "__main__":
    main()
