import gradio as gr
from PIL import Image
from utils import encode_image_to_datauri, cot_with_gpt, extract_instructions, infer_with_DiT, roi_localization, fusion
import openai
import os
import uuid
from src.flux.generate import generate, seed_everything


def process_image(api_key, seed, image, prompt):
    if not api_key:
        raise gr.Error("❌ Please enter a valid OpenAI API key.")
    
    openai.api_key = api_key

    # Generate a unique image ID to avoid file name conflict
    image_id = str(uuid.uuid4())
    seed_everything(seed)
    input_path = f"input_{image_id}.png"
    image.save(input_path)

    try:
        uri = encode_image_to_datauri(input_path)
        categories, instructions = cot_with_gpt(uri, prompt)
        # categories = ['Tone Transfer', 'Style Change']
        # instructions = ['Change the time to night', 'Change the style to watercolor']

        if not categories or not instructions:
            raise gr.Error("No editing steps returned by GPT. Try a more specific instruction.")

        intermediate_images = []
        current_image_path = input_path

        for i, (category, instruction) in enumerate(zip(categories, instructions)):
            print(f"[Step {i}] Category: {category} | Instruction: {instruction}")
            step_prefix = f"{image_id}_{i}"

            if category in ('Add', 'Remove', 'Replace'):
                if category == 'Add':
                    edited_image = infer_with_DiT('RoI Editing', current_image_path, instruction, category)
                else:
                    mask_image = roi_localization(current_image_path, instruction, category)
                    edited_image = infer_with_DiT('RoI Inpainting', mask_image, instruction, category)

            elif category == 'Action Change':
                mask_image = roi_localization(current_image_path, instruction, category)
                inpainted = infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove')
                changed_instance, x0, y1, scale = infer_with_DiT('RoI Editing', current_image_path, instruction, category)
                fusion_image = fusion(inpainted, changed_instance, x0, y1, scale)
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)

            elif category in ('Move', 'Resize'):
                mask_image, changed_instance, x0, y1, scale = roi_localization(current_image_path, instruction, category)
                inpainted = infer_with_DiT('RoI Inpainting', mask_image, instruction, 'Remove')
                fusion_image = fusion(inpainted, changed_instance, x0, y1, scale)
                edited_image = infer_with_DiT('RoI Compositioning', fusion_image, instruction, None)

            elif category in ('Appearance Change', 'Background Change', 'Color Change', 'Material Change', 'Expression Change'):
                edited_image = infer_with_DiT('RoI Editing', current_image_path, instruction, category)

            elif category in ('Tone Transfer', 'Style Change'):
                edited_image = infer_with_DiT('Global Transformation', current_image_path, instruction, category)

            else:
                raise gr.Error(f"Invalid category returned: '{category}'")

            current_image_path = f"{step_prefix}.png"
            edited_image.save(current_image_path)
            intermediate_images.append(edited_image.copy())

        final_result = intermediate_images[-1] if intermediate_images else image
        return intermediate_images, final_result

    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ IEAP: Image Editing As Programs")

    with gr.Row():
        api_key_input = gr.Textbox(label="🔑 OpenAI API Key", type="password", placeholder="sk-...")
    
    with gr.Row():
        seed_slider = gr.Slider(
            label="🎲 Random Seed",
            minimum=0,
            maximum=1000000,
            value=3407,
            step=1,
            info="Drag to set the random seed for reproducibility"
        )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(label="Instruction", placeholder="e.g., Move the dog to the left and change its color to blue")
            submit_button = gr.Button("Submit")
        with gr.Column():
            result_gallery = gr.Gallery(label="Intermediate Steps", columns=2, height="auto")
            final_output = gr.Image(label="✅ Final Result")

    submit_button.click(
        fn=process_image,
        inputs=[api_key_input, seed_slider, image_input, prompt_input],
        outputs=[result_gallery, final_output]
    )

if __name__ == "__main__":
    demo.launch(    
    )
