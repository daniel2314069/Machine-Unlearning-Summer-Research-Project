import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["W_0", "W"], required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if len(prompts) != 10:
        raise ValueError(f"Expected 10 prompts, got {len(prompts)}")
    forbidden = ("cat", "kitten", "feline")
    for prompt in prompts:
        words = {word.strip(".,;:!?\"'").lower() for word in prompt.split()}
        overlap = words.intersection(forbidden)
        if overlap:
            raise ValueError(f"Forbidden keyword {overlap} in: {prompt}")

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
    ).to(args.device)
    if args.variant == "W":
        if args.weights is None:
            raise ValueError("--weights is required for W")
        pipe.unet.load_state_dict(load_file(str(args.weights)), strict=False)

    variant_dir = args.output_dir / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        seed = args.base_seed + index - 1
        generator = torch.Generator(device=args.device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(variant_dir / f"{index:02d}_seed_{seed}.png")
        print(f"[{args.variant}] {index:02d}/10 seed={seed}")


if __name__ == "__main__":
    main()
