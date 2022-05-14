import requests
import io
import os
from encoders.x_transformer import default
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from encoders.modules import BERTEmbedder
import clip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow *
                       (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety +
                           size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


class Args:
    def __init__(self, model_variant="base"):
        if model_variant == "jack":
            self.model_path = "checkpoints/finetune.pt"
        elif model_variant == "base":
            self.model_path = "checkpoin5s/base.pt"
        else:
            self.model_path = "checkpoints/inpaint.p"
        self.kl_path = "checkpoints/kl-f8.pt"
        self.bert_path = "checkpoints/bert.pt"
        self.text = ''
        self.edit = ''
        self.edit_x = 0
        self.edit_y = 0
        self.edit_width = 256
        self.edit_height = 256
        self.mask = ''
        self.negative = ''
        self.init_image = None
        self.skip_timesteps = 0
        self.prefix = ''
        self.num_batches = 1
        self.batch_size = 1
        self.width = 256
        self.height = 256
        self.seed = -1
        self.guidance_scale = 5.0
        self.steps = 25
        self.cpu = False
        self.clip_model = 'ViT-L/14'
        self.clip_score = False
        self.clip_guidance = False
        self.clip_guidance_scale = 150
        self.cutn = 16
        self.ddim = False
        self.ddpm = False

    def __str__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def default_model_params():
    return {
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '27',  # Modify this value to decrease the number of
        # timesteps.
        'image_size': 32,
        'learn_sigma': False,
        'noise_schedule': 'linear',
        'num_channels': 320,
        'num_heads': 8,
        'num_res_blocks': 2,
        'resblock_updown': False,
        'use_fp16': False,
        'use_scale_shift_norm': False,
        'clip_embed_dim': None,
        'image_condition': False,
        'super_res_condition': False
    }


def get_model_params(model_state_dict, args):
    model_params = default_model_params()

    if 'clip_proj.weight' in model_state_dict:
        model_params['clip_embed_dim'] = 768

    if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8:
        model_params['image_condition'] = True

    if 'external_block.0.0.weight' in model_state_dict:
        model_params['super_res_condition'] = True

    if args.ddpm:
        model_params['timestep_respacing'] = 1000
    if args.ddim:
        if args.steps:
            model_params['timestep_respacing'] = 'ddim'+str(args.steps)
        else:
            model_params['timestep_respacing'] = 'ddim50'
    elif args.steps:
        model_params['timestep_respacing'] = str(args.steps)

    if args.cpu:
        model_params['use_fp16'] = False
    return model_params


def load_ldm(args):
    # vae
    ldm = torch.load(args.kl_path, map_location="cpu")
    ldm.to(device)
    ldm.eval()
    ldm.requires_grad_(args.clip_guidance)
    set_requires_grad(ldm, args.clip_guidance)
    return ldm


def load_bert(args):
    bert = BERTEmbedder(1280, 32)
    sd = torch.load(args.bert_path, map_location="cpu")
    bert.load_state_dict(sd)

    bert.to(device)
    bert.half().eval()
    set_requires_grad(bert, False)
    return bert


def load_clip_model(args):
    clip_model, clip_preprocess = clip.load(
        args.clip_model, device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    return clip_model, clip_preprocess


def load_models(args):
    if args.edit and not args.mask:
        from draw import Draw

    model_state_dict = torch.load(args.model_path, map_location='cpu')
    model_params = get_model_params(model_state_dict, args)

    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)

    # Load models
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(model_state_dict, strict=False)
    model.requires_grad_(args.clip_guidance).eval().to(device)

    if model_config['use_fp16']:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()

    ldm = load_ldm(args)
    bert = load_bert(args)

    # clip
    clip_model, clip_preprocess = load_clip_model(args)

    models = {
        "model": model,
        "diffusion": diffusion,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "bert": bert,
        "ldm": ldm
    }

    return model_params, models


def bert_encoding(bert, text, args):
    return bert.encode([text]*args.batch_size).to(device).float()


def do_run(args, model_params, models):
    model = models["model"]
    diffusion = models["diffusion"]
    clip_model = models["clip_model"]
    clip_preprocess = models["clip_preprocess"]
    bert = models["bert"]
    ldm = models["ldm"]

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    # bert context
    text_emb = bert_encoding(bert, args.text, args)
    text_blank = bert_encoding(bert, args.negative, args)

    # clip encodings
    text = clip.tokenize([args.text]*args.batch_size, truncate=True).to(device)
    text_clip_blank = clip.tokenize(
        [args.negative]*args.batch_size, truncate=True).to(device)
    # clip context
    text_emb_clip = clip_model.encode_text(text)
    text_emb_clip_blank = clip_model.encode_text(text_clip_blank)

    text_emb_norm = text_emb_clip[0] / \
        text_emb_clip[0].norm(dim=-1, keepdim=True)

    image_embed = None
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[
        0.26862954, 0.26130258, 0.27577711])
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn)

    # image context
    if args.edit:
        if args.edit.endswith('.npy'):
            with open(args.edit, 'rb') as f:
                im = np.load(f)
                im = torch.from_numpy(im).unsqueeze(0).to(device)

                input_image = torch.zeros(
                    1, 4, args.height//8, args.width//8, device=device)

                y = args.edit_y//8
                x = args.edit_x//8

                ycrop = y + im.shape[2] - input_image.shape[2]
                xcrop = x + im.shape[3] - input_image.shape[3]

                ycrop = ycrop if ycrop > 0 else 0
                xcrop = xcrop if xcrop > 0 else 0

                input_image[0, :, y if y >= 0 else 0:y+im.shape[2], x if x >= 0 else 0:x+im.shape[3]
                            ] = im[:, :, 0 if y > 0 else -y:im.shape[2]-ycrop, 0 if x > 0 else -x:im.shape[3]-xcrop]

                input_image_pil = ldm.decode(input_image)
                input_image_pil = TF.to_pil_image(
                    input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

                input_image *= 0.18215
        else:
            w = args.edit_width if args.edit_width else args.width
            h = args.edit_height if args.edit_height else args.height

            input_image_pil = Image.open(fetch(args.edit)).convert('RGB')
            input_image_pil = ImageOps.fit(input_image_pil, (w, h))

            input_image = torch.zeros(
                1, 4, args.height//8, args.width//8, device=device)

            im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
            im = 2*im-1
            im = ldm.encode(im).sample()

            y = args.edit_y//8
            x = args.edit_x//8

            input_image = torch.zeros(
                1, 4, args.height//8, args.width//8, device=device)

            ycrop = y + im.shape[2] - input_image.shape[2]
            xcrop = x + im.shape[3] - input_image.shape[3]

            ycrop = ycrop if ycrop > 0 else 0
            xcrop = xcrop if xcrop > 0 else 0

            input_image[0, :, y if y >= 0 else 0:y+im.shape[2], x if x >= 0 else 0:x+im.shape[3]
                        ] = im[:, :, 0 if y > 0 else -y:im.shape[2]-ycrop, 0 if x > 0 else -x:im.shape[3]-xcrop]

            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(
                input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

            input_image *= 0.18215

        if args.mask:
            mask_image = Image.open(fetch(args.mask)).convert('L')
            mask_image = mask_image.resize(
                (args.width//8, args.height//8), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
        else:
            print('draw the area for inpainting, then close the window')
            app = QApplication(sys.argv)
            d = Draw(args.width, args.height, input_image_pil)
            app.exec_()
            mask_image = d.getCanvas().convert('L').point(lambda p: 255 if p < 1 else 0)
            mask_image.save('mask.png')
            mask_image = mask_image.resize(
                (args.width//8, args.height//8), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

        mask1 = (mask > 0.5)
        mask1 = mask1.float()

        input_image *= mask1

        image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()
    elif model_params['image_condition']:
        # using inpaint model but no image is provided
        image_embed = torch.zeros(
            args.batch_size*2, 4, args.height//8, args.width//8, device=device)

    kwargs = {
        "context": torch.cat([text_emb, text_blank], dim=0).float(),
        "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float() if model_params['clip_embed_dim'] else None,
        "image_embed": image_embed
    }

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    cur_t = None

    def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
        with torch.enable_grad():
            x = x[:args.batch_size].detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            kw = {
                'context': context[:args.batch_size],
                'clip_embed': clip_embed[:args.batch_size] if model_params['clip_embed_dim'] else None,
                'image_embed': image_embed[:args.batch_size] if image_embed is not None else None
            }

            out = diffusion.p_mean_variance(
                model, x, my_t, clip_denoised=False, model_kwargs=kw)

            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_in /= 0.18215

            x_img = ldm.decode(x_in)

            clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(
                clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])

            losses = dists.sum(2).mean(0)

            loss = losses.sum() * args.clip_guidance_scale

            return -torch.autograd.grad(loss, x)[0]

    if args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def decode_image(sample):
        images = []
        for image in sample['pred_xstart'][:args.batch_size]:
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)
            res = {
                'npy': image.detach().cpu().numpy(),
                'pil': TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
            }
            images.append(res)
        return images

    def save_sample(i, sample, clip_score=False):
        os.makedirs("output", exist_ok=True)
        os.makedirs("output_npy", exist_ok=True)
        for k, image_o in enumerate(images):
            img_npy = image_o['npy']
            img_pil = image_o['pil']
            npy_filename = f'output_npy/{args.prefix}{i * args.batch_size + k:05}.npy'
            with open(npy_filename, 'wb') as outfile:
                np.save(outfile, img_npy)

            filename = f'output/{args.prefix}{i * args.batch_size + k:05}.png'
            img_pil.save(filename)

            if clip_score:
                image_emb = clip_model.encode_image(
                    clip_preprocess(img_pil).unsqueeze(0).to(device))
                image_emb_norm = image_emb / \
                    image_emb.norm(dim=-1, keepdim=True)

                similarity = torch.nn.functional.cosine_similarity(
                    image_emb_norm, text_emb_norm, dim=-1)

                final_filename = f'output/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.png'
                os.rename(filename, final_filename)

                npy_final = f'output_npy/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.npy'
                os.rename(npy_filename, npy_final)

    if args.init_image:
        init = args.init_image.resize(
            (int(args.width),  int(args.height)), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).clamp(0, 1)
        h = ldm.encode(init * 2 - 1).sample() * 0.18215
        init = torch.cat(args.batch_size*2*[h], dim=0)
    else:
        init = None

    res_images = []
    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=cond_fn if args.clip_guidance else None,
            device=device,
            progress=True,
            init_image=init,
            skip_timesteps=args.skip_timesteps,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                images = decode_image(sample)
                save_sample(i, sample)

        images = decode_image(sample)
        save_sample(i, sample, args.clip_score)
        return images
