
import diffusers
import torch
import diffusers.models.embeddings as embeddings
import diffusers.pipelines.stable_diffusion as stable_diffusion
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, CLIPTextModel, CLIPVisionModel


class StableDiffusion():
    def __init__(self,
                 model_id = "stabilityai/stable-diffusion-2-1-base",
                 scheduler = "EulerDiscreteScheduler",
                 ):
        scheduler = getattr(diffusers, scheduler).from_pretrained(model_id, subfolder="scheduler")
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.bfloat16)
        self.device = self.pipe.device
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.feature_extractor = self.pipe.feature_extractor
        self.image_encoder = self.pipe.image_encoder
        self.image_processor = self.pipe.image_processor


    def encode_text(self, text):
        if text is not None and isinstance(text, str):
            batch_size = 1
        elif text is not None and isinstance(text, list):
            batch_size = len(text)

        text_inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(text, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None
        
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0].to(dtype=self.unet.dtype, device=self.unet.device)
      
        return prompt_embeds
    
    def encode_image(self, image):
        latent = self.vae.encode(image, return_dict=False)[0].mode()
        latent = latent * self.vae.config.scaling_factor
        return latent
        

    def decode_latents(self, latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        return image
    
    def postprocess_image(self, image):
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type="np", do_denormalize=do_denormalize)
        return image


    def to(self, device):
        self.pipe.to(device)
        self.device = device
        return self
    
    @torch.no_grad()
    def __call__(self,  prompt, num_inference_steps=50):
        batch_size = len(prompt)
        
        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor

        # 1. extract text embeddings
        text_embeds = self.encode_text(prompt)
        print(text_embeds.shape)
        # 2. setup scheduler
        timesteps, num_inference_steps = stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps(self.scheduler, num_inference_steps, self.device)
        print(timesteps.shape)

        # 3. prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeds.dtype,
            self.device,
            None,
        )
        print(latents.shape)

        # 4. run diffusion
        for i, t in enumerate(timesteps):

            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds,
                return_dict=False,
            )[0]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,  return_dict=False)[0]

        

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        print(image.shape)
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type="np", do_denormalize=do_denormalize)

        return image