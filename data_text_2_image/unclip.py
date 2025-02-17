import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from diffusers import UnCLIPPipeline

class UnClip(nn.Module):

    def __init__(self, 
                device: str = "cpu",
                dtype: torch.dtype = torch.float32,
                ):
        self.pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=dtype)
        self.pipe = self.pipe.to(device)  
        self.device = device  

    def to(self, device):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def text2emb(self,
                prompt: str = None,
                num_images_per_prompt: int = 1,
                prior_num_inference_steps: int = 25,
                prior_guidance_scale: float = 4.0,
            ):
            batch_size = 1
            if isinstance(prompt, list):
                batch_size = len(prompt)

            batch_size = batch_size * num_images_per_prompt

            do_classifier_free_guidance = prior_guidance_scale > 1.0

            prompt_embeds, text_enc_hid_states, text_mask = self.pipe._encode_prompt(
                prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, None, None
            )

            # prior

            self.pipe.prior_scheduler.set_timesteps(prior_num_inference_steps, device=self.device)
            prior_timesteps_tensor = self.pipe.prior_scheduler.timesteps

            embedding_dim = self.pipe.prior.config.embedding_dim

            prior_latents = self.pipe.prepare_latents(
                (batch_size, embedding_dim),
                prompt_embeds.dtype,
                self.device,
                None, 
                None,
                self.pipe.prior_scheduler,
            )

            for i, t in enumerate(prior_timesteps_tensor):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

                predicted_image_embedding = self.pipe.prior(
                    latent_model_input,
                    timestep=t,
                    proj_embedding=prompt_embeds,
                    encoder_hidden_states=text_enc_hid_states,
                    attention_mask=text_mask,
                ).predicted_image_embedding

                if do_classifier_free_guidance:
                    predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                    predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                        predicted_image_embedding_text - predicted_image_embedding_uncond
                    )

                if i + 1 == prior_timesteps_tensor.shape[0]:
                    prev_timestep = None
                else:
                    prev_timestep = prior_timesteps_tensor[i + 1]

                prior_latents = self.pipe.prior_scheduler.step(
                    predicted_image_embedding,
                    timestep=t,
                    sample=prior_latents,
                    prev_timestep=prev_timestep,
                ).prev_sample

            prior_latents = self.pipe.prior.post_process_latents(prior_latents)

            image_embeddings = prior_latents

            return image_embeddings, prompt_embeds, text_enc_hid_states, text_mask
    
    def emb2img(self, 
                image_embeddings, 
                prompt_embeds,  
                text_enc_hid_states,
                text_mask,   
                decoder_num_inference_steps: int = 25,
                decoder_guidance_scale: float = 8.0,
                ):
        do_classifier_free_guidance = decoder_guidance_scale > 1.0
        
        text_enc_hid_states, additive_clip_time_embeddings = self.pipe.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_enc_hid_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        batch_size = text_enc_hid_states.shape[0]//2

        decoder_text_mask = F.pad(text_mask, (self.pipe.text_proj.clip_extra_context_tokens, 0), value=True)

        self.pipe.decoder_scheduler.set_timesteps(decoder_num_inference_steps, device=self.device)
        decoder_timesteps_tensor = self.pipe.decoder_scheduler.timesteps

        num_channels_latents = self.pipe.decoder.config.in_channels
        height = self.pipe.decoder.config.sample_size
        width = self.pipe.decoder.config.sample_size

        decoder_latents = self.pipe.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_enc_hid_states.dtype,
            self.device,
            None,
            None,
            self.pipe.decoder_scheduler,
        )

        for i, t in enumerate(decoder_timesteps_tensor):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.pipe.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_enc_hid_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.pipe.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=None
            ).prev_sample

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents
        return image_small

    def sr(self, image_small,
            super_res_num_inference_steps: int = 25,
            ):
        batch_size = image_small.shape[0]

        self.pipe.super_res_scheduler.set_timesteps(super_res_num_inference_steps, device=self.device)
        super_res_timesteps_tensor = self.pipe.super_res_scheduler.timesteps

        channels = self.pipe.super_res_first.config.in_channels // 2
        height = self.pipe.super_res_first.config.sample_size
        width = self.pipe.super_res_first.config.sample_size

        super_res_latents = self.pipe.prepare_latents(
            (batch_size, channels, height, width),
            image_small.dtype,
            self.device,
            None,
            None,
            self.pipe.super_res_scheduler,
        )


        interpolate_antialias = {}
        if "antialias" in inspect.signature(F.interpolate).parameters:
            interpolate_antialias["antialias"] = True

        image_upscaled = F.interpolate(
            image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
        )

        for i, t in enumerate(super_res_timesteps_tensor):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.pipe.super_res_last
            else:
                unet = self.pipe.super_res_first

            latent_model_input = torch.cat([super_res_latents, image_upscaled], dim=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            ).sample

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.pipe.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=None
            ).prev_sample

        image = super_res_latents

        return image

         