import gc
import argparse

from sample_api import *


# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='checkpoints/inpaint.pt',
                    help='path to the diffusion model')

parser.add_argument('--kl_path', type=str, default='checkpoints/kl-f8.pt',
                    help='path to the LDM first stage model')

parser.add_argument('--bert_path', type=str, default='checkpoints/bert.pt',
                    help='path to the LDM first stage model')

parser.add_argument('--text', type=str, required=False, default='pizza',
                    help='your text prompt')

parser.add_argument('--edit', type=str, required=False,
                    help='path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)', default='/home/ubuntu/vangap/glid-3-xl/latent-diffusion/glid-3-xl/output/edit.png')

parser.add_argument('--edit_x', type=int, required=False, default=0,
                    help='x position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_y', type=int, required=False, default=0,
                    help='y position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_width', type=int, required=False, default=0,
                    help='width of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_height', type=int, required=False, default=0,
                    help='height of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--mask', type=str, required=False,
                    help='path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8', default='/home/ubuntu/vangap/glid-3-xl/latent-diffusion/glid-3-xl/image.jpeg')

parser.add_argument('--negative', type=str, required=False, default='',
                    help='negative text prompt')

parser.add_argument('--init_image', type=str, required=False, default=None,
                    help='init image to use')

parser.add_argument('--skip_timesteps', type=int, required=False, default=0,
                    help='how many diffusion steps are gonna be skipped')

parser.add_argument('--prefix', type=str, required=False, default='',
                    help='prefix for output files')

parser.add_argument('--num_batches', type=int, default=1, required=False,
                    help='number of batches')

parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='batch size')

parser.add_argument('--width', type=int, default=256, required=False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--height', type=int, default=256, required=False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type=int, default=-1, required=False,
                    help='random seed')

parser.add_argument('--guidance_scale', type=float, default=5.0, required=False,
                    help='classifier-free guidance scale')

parser.add_argument('--steps', type=int, default=0, required=False,
                    help='number of diffusion steps')

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--clip_model', dest='clip_model', default='ViT-L/14')

parser.add_argument('--clip_score', dest='clip_score', action='store_true')

parser.add_argument('--clip_guidance',
                    dest='clip_guidance', action='store_true')

parser.add_argument('--clip_guidance_scale', type=float, default=150, required=False,
                    help='Controls how much the image should look like the prompt')  # may need to use lower value for ddim

parser.add_argument('--cutn', type=int, default=16, required=False,
                    help='Number of cuts')

# turn on to use 50 step ddim
parser.add_argument('--ddim', dest='ddim', action='store_true')

# turn on to use 50 step ddim
parser.add_argument('--ddpm', dest='ddpm', action='store_true')

args = parser.parse_args()

model_params, models = load_models(args)
gc.collect()
do_run(args, model_params, models)
