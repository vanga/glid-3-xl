# export FLASK_APP=flask-server.py
# flask run --host="0.0.0.0" --port=80
from random import randint, choice
import json
import torch
import numpy as np
from PIL import Image
import base64
import cv2
import argparse
from io import BytesIO

from flask import Flask
from flask import request

from sample_api import *


app = Flask(__name__)
device = torch.device('cuda:0')

args = Args("jack")
model_params, models = load_models(args)


def glid_3_xl_generate(req_body):
    print(json.dumps(req_body))

    args.text = req_body['prompt']
    print(f"Generating images for '{args.text}'")
    args.negative_prompt = req_body['negative_prompt']
    args.width = min(req_body['width'], 512)
    args.height = min(req_body['height'], 512)
    args.skip_timesteps = min(req_body['skip_timesteps'], 25)
    args.guidance_scale = min(req_body['guidance_scale'], 15)
    args.batch_size = min(req_body['batch_size'], 4)

    if req_body['init_image']:
        print("processing init image")
        im_bytes = base64.b64decode(req_body['init_image'])
        img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
        args.init_image = img
    else:
        req_body['init_image'] = None

    images = do_run(args, model_params, models)
    res_images = []
    for image in images:
        buffered = BytesIO()
        image['pil'].save(buffered, format="JPEG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        res_images.append(image_b64)
    res_body = {
        "images": res_images
    }

    return json.dumps(res_body)


@app.route('/glid-3-xl-jack', methods=['POST'])
def glid_3_xl_jack():
    req_body = request.get_json(force=True)
    return glid_3_xl_generate(req_body)

@app.route('/health')
def health():
    return "OK"
