import os

from PIL import Image


def upscale_images_in_folder_bicubic(
    input_folder_path,
    output_folder_path,
    target_size=(512, 512),
    expected_input_size=(128, 128),
):
    """
    Upsamples images in a folder using bicubic interpolation.

    Args:
        input_folder_path (str): Path to the folder containing .png images.
        output_folder_path (str): Path to the folder where upscaled images will be saved.
        target_size (tuple): Desired output resolution (width, height).
        expected_input_size (tuple): Expected input resolution (width, height) for optional check.
    """
    print("Using bicubic interpolation for upscaling.")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")

    # List all files in the input folder
    try:
        image_files = [
            f
            for f in os.listdir(input_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except FileNotFoundError:
        print(f"Error: Input folder not found at {input_folder_path}")
        return
    except Exception as e:
        print(f"Error accessing input folder: {e}")
        return

    if not image_files:
        print(f"No .png, .jpg, or .jpeg images found in {input_folder_path}")
        return

    print(f"Found {len(image_files)} images to upscale.")

    for image_name in image_files:
        input_image_path = os.path.join(input_folder_path, image_name)
        output_image_path = os.path.join(output_folder_path, image_name)

        print(f"\nProcessing {image_name}...")

        try:
            # Open the low-resolution image
            img = Image.open(input_image_path).convert("RGB")

            # Verify image resolution (optional, but good for sanity check)
            if img.size != expected_input_size:
                print(
                    f"Warning: Image {image_name} is not {expected_input_size[0]}x{expected_input_size[1]} (found {img.size}). "
                    f"It will still be upscaled to {target_size[0]}x{target_size[1]}."
                )
            # If you want to force resize to expected_input_size first (e.g., 128x128), uncomment below:
            # if img.size != expected_input_size:
            #     print(f"Resizing {image_name} from {img.size} to {expected_input_size} before upscaling.")
            #     img = img.resize(expected_input_size, Image.Resampling.LANCZOS) # Or BICUBIC if preferred for pre-resize

            # Upscale the image using bicubic interpolation
            print(
                f"Upscaling {image_name} to {target_size[0]}x{target_size[1]} using bicubic interpolation..."
            )
            upscaled_image = img.resize(target_size, Image.Resampling.BICUBIC)

            # Save the upscaled image
            upscaled_image.save(output_image_path)
            print(f"Saved upscaled image to: {output_image_path}")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue  # Skip to the next image if an error occurs

    print("\nImage upscaling process complete.")


if __name__ == "__main__":
    # Define the input and output folder paths
    input_folder = "/pub4/qasim/1xgpt/humanoid_world_model/submissions/diffusion128x128"
    output_folder = "/pub4/qasim/1xgpt/humanoid_world_model/submissions/bicubic512x512"  # Changed output folder name

    # Define the target size for upscaling
    output_resolution = (512, 512)
    # Define the expected input size (for the warning message)
    input_resolution_check = (128, 128)

    print("Starting image upscaling script using bicubic interpolation...")
    upscale_images_in_folder_bicubic(
        input_folder,
        output_folder,
        target_size=output_resolution,
        expected_input_size=input_resolution_check,
    )
    print("Script finished.")

# import os

# import torch
# from diffusers import StableDiffusionUpscalePipeline
# from PIL import Image


# def upscale_images_in_folder(
#     input_folder_path,
#     output_folder_path,
#     model_id="stabilityai/stable-diffusion-x4-upscaler",
#     prompt="an upscaled image",
# ):
#     """
#     Upsamples images in a folder from 128x128 to 512x512 using Stable Diffusion x4 upsampler.

#     Args:
#         input_folder_path (str): Path to the folder containing 128x128 .png images.
#         output_folder_path (str): Path to the folder where 512x512 upscaled images will be saved.
#         model_id (str): The model ID for the Stable Diffusion upscaler on Hugging Face.
#         prompt (str): A general prompt to guide the upscaling process.
#     """
#     # Check if CUDA is available and set device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Load model and scheduler
#     try:
#         pipeline = StableDiffusionUpscalePipeline.from_pretrained(
#             model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
#         )
#         pipeline = pipeline.to(device)
#     except Exception as e:
#         print(f"Error loading the model: {e}")
#         print(
#             "Please ensure you have the necessary libraries installed and are logged into Hugging Face if required for the model."
#         )
#         print("You might need to run: huggingface-cli login")
#         return

#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#         print(f"Created output folder: {output_folder_path}")

#     # List all files in the input folder
#     try:
#         image_files = [
#             f for f in os.listdir(input_folder_path) if f.lower().endswith(".png")
#         ]
#     except FileNotFoundError:
#         print(f"Error: Input folder not found at {input_folder_path}")
#         return
#     except Exception as e:
#         print(f"Error accessing input folder: {e}")
#         return

#     if not image_files:
#         print(f"No .png images found in {input_folder_path}")
#         return

#     print(f"Found {len(image_files)} .png images to upscale.")

#     for image_name in image_files:
#         input_image_path = os.path.join(input_folder_path, image_name)
#         output_image_path = os.path.join(output_folder_path, image_name)

#         print(f"\nProcessing {image_name}...")

#         try:
#             # Open the low-resolution image
#             low_res_img = Image.open(input_image_path).convert("RGB")

#             # Verify image resolution (optional, but good for sanity check)
#             if low_res_img.size != (128, 128):
#                 print(
#                     f"Warning: Image {image_name} is not 128x128 (found {low_res_img.size}). Resizing to 128x128 before upscaling."
#                 )
#                 low_res_img = low_res_img.resize((128, 128))

#             # Upscale the image
#             print(f"Upscaling {image_name} with prompt: '{prompt}'")
#             upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

#             # Save the upscaled image
#             upscaled_image.save(output_image_path)
#             print(f"Saved upscaled image to: {output_image_path}")

#         except Exception as e:
#             print(f"Error processing image {image_name}: {e}")
#             continue  # Skip to the next image if an error occurs

#     print("\nImage upscaling process complete.")


# if __name__ == "__main__":
#     # Define the input and output folder paths
#     input_folder = "/pub4/qasim/1xgpt/humanoid_world_model/submissions/diffusion128x128"
#     output_folder = (
#         "/pub4/qasim/1xgpt/humanoid_world_model/submissions/diffusion512x512"
#     )

#     # You can customize the prompt if needed for your specific images
#     # For example, if all images are of a "humanoid robot", you could use that.
#     # Using a generic prompt as the content of the images is unknown.
#     custom_prompt = "a high-resolution, detailed image"

#     print("Starting image upscaling script...")
#     upscale_images_in_folder(input_folder, output_folder, prompt=custom_prompt)
#     print("Script finished.")
