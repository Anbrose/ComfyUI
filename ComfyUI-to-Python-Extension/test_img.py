import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import StreamingResponse
import shutil
import torch
import asyncio
import io
import re

# FastAPI app
app = FastAPI()

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_max_numbered_file(directory):
    max_number = -1
    max_file = None

    # 正则表达式用于匹配文件名中的数字
    pattern = re.compile(r'ComfyUITest_(\d+)_\.png')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_file = filename

    return max_file

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    ControlNetLoader,
    SaveImage,
    KSampler,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    LoraLoader,
    VAEDecode,
    LoadImage,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
)


def process_image(image_path: str):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="architecturerealmix_v11.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_17 = loraloader.load_lora(
            lora_name="mjmimic.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="4K,cinematic, masterpiece, Photo of a dramatic landscape with crepuscular rays piercing through the sky, rim light, dusk, modern building, blue sky, bright theme",
            clip=get_value_at_index(loraloader_17, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="owres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpg artifacts,signature,watermark,username,blurry, bad building, bad cars, bad river, bad lak",
            clip=get_value_at_index(loraloader_17, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_11 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_canny.safetensors"
        )

        loadimage = LoadImage()
        loadimage_12 = loadimage.load_image(image=image_path)

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_14 = emptylatentimage.generate(
            width=768, height=512, batch_size=1
        )

        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        cannyedgepreprocessor_16 = cannyedgepreprocessor.execute(
            low_threshold=100,
            high_threshold=200,
            resolution=512,
            image=get_value_at_index(loadimage_12, 0),
        )

        controlnetapplyadvanced_13 = controlnetapplyadvanced.apply_controlnet(
            strength=1,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            control_net=get_value_at_index(controlnetloader_11, 0),
            image=get_value_at_index(cannyedgepreprocessor_16, 0),
        )

        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=27,
            cfg=5,
            sampler_name="dpmpp_2m_sde",
            scheduler="normal",
            denoise=0.99,
            model=get_value_at_index(loraloader_17, 0),
            positive=get_value_at_index(controlnetapplyadvanced_13, 0),
            negative=get_value_at_index(controlnetapplyadvanced_13, 1),
            latent_image=get_value_at_index(emptylatentimage_14, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        saveimage_19 = saveimage.save_images(
            filename_prefix="ComfyUITest", images=get_value_at_index(vaedecode_8, 0)
        )


@app.post("/process-image/")
async def process_upload_image(
        file: UploadFile = File(...)):
    temp_file_path = f"/ComfyUI/input/temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_image(temp_file_path)
    processed_image_path = find_max_numbered_file("/ComfyUI/output/")

    # Read the processed image into memory
    with open("/ComfyUI/output/"+processed_image_path, "rb") as f:
        image_data = f.read()

    # Clean up temporary files

    # Return the image as a response
    print("Read image from {}".format(temp_file_path))
    return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
