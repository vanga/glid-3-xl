# import ipywidgets as widgets
import shutil
import glob
import gradio as gr
import argparse
import os

from sample_api import *

# base_dir = ""
output_dir = "./output"

args = Args("jack")

model_params, models = load_models(args)


def adv_run(prompt, negative, init_image, skips, guidance, amount_per_batch, width, height):

    args.text = prompt
    args.negative = negative
    args.init_image = init_image
    args.skip_timesteps = skips
    args.guidance_scale = guidance
    args.batch_size = amount_per_batch
    args.width = width
    args.height = height
    shutil.rmtree(output_dir, True)
    os.makedirs(output_dir, exist_ok=True)
    do_run(args, model_params, models)
    output_images = []

    for file in tqdm(glob.glob(f"{output_dir}/*.png")):
        output_images.append(Image.open(file))
        #         swinUpscale(file, show_large)
        # !montage - geometry + 1+1 - background black $base_dir/output/*.png $base_dir/grid.png
        # return Image.open(f"{base_dir}/grid.png")
    return output_images


iface = gr.Interface(
    fn=adv_run,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="text cue for generating images",
                          default="", label=None, optional=False),
        "text",
        gr.inputs.Image(shape=(256, 256), optional=True,
                        type="pil", label='Starting Image'),
        gr.inputs.Slider(
            0, 25, 1, default=0, label="Image guidance strength (Must be used when using image cues)"),
        gr.inputs.Slider(1, 15, 1, default=5,
                         label="Text cue guidance strength"),
        # gr.inputs.Slider(1, 32, 1, default=1, label='Number of batches')
        gr.inputs.Slider(
            1, 16, 1, default=1, label='Number of images to generate'),
        gr.inputs.Slider(16, 512, 16, default=256),
        gr.inputs.Slider(16, 512, 16, default=256),
        # gr.inputs.Checkbox(
        #     default=False, label="Clip Rerank (for batch images)", optional=False),
        # gr.inputs.Checkbox(
        #     default=True, label="Increase sharpness using SwinIR", optional=False),
        # gr.inputs.Checkbox(
        # default=False, label="Show SwinIR results as 512x512 (less sharp)", optional=False)
    ],
    outputs=gr.outputs.Carousel(['image']))

iface.launch(share=False, debug=True, inline=False,
             enable_queue=True, server_port=12344, server_name="0.0.0.0")
