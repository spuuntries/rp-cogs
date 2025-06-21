from cog import BasePredictor, Input, Path, ConcatenateIterator
import os
import re
import time
import torch
import subprocess
import numpy as np
import timm
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from einops import rearrange

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://huggingface.co/spuun/fp-nlp/resolve/main/model-fp-nlp-2025-06-19_08-31-17.tar?download=true"  # Update with your model URL


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionProjection(nn.Module):
    def __init__(self, vision_dim, llama_dim, num_groups=4):
        super().__init__()
        hidden_dim = llama_dim * 4
        self.num_groups = num_groups
        self.target_channels = vision_dim  # 2048

        # Create 1x1 conv layers for different ResNet channel sizes
        self.channel_adapters = nn.ModuleDict(
            {
                "64": nn.Conv2d(64, self.target_channels, 1),
                "128": nn.Conv2d(128, self.target_channels, 1),
                "256": nn.Conv2d(256, self.target_channels, 1),
                "512": nn.Conv2d(512, self.target_channels, 1),
                "1024": nn.Conv2d(1024, self.target_channels, 1),
                "2048": nn.Conv2d(2048, self.target_channels, 1),
            }
        )

        concat_dim = self.target_channels * (num_groups + 1)
        self.mlp = MLP(concat_dim, hidden_dim, llama_dim)

    def dense_channel_integration(self, multi_layer_features_batch):
        batch_results = []
        target_size = (7, 7)

        for sample_features in multi_layer_features_batch:
            if not sample_features:
                batch_results.append(
                    torch.zeros(
                        1, self.num_groups + 1, self.target_channels, *target_size
                    )
                )
                continue

            uniform_layers = []
            for layer_feat in sample_features:
                # Ensure 4D: add batch dimension if missing
                if len(layer_feat.shape) == 3:
                    layer_feat = rearrange(layer_feat, "c h w -> 1 c h w")

                # Resize spatial dimensions first
                spatially_resized = F.adaptive_avg_pool2d(layer_feat, target_size)

                # Use learnable 1x1 conv to adjust channels
                B, C, H, W = spatially_resized.shape

                if str(C) in self.channel_adapters:
                    channel_adjusted = self.channel_adapters[str(C)](spatially_resized)
                else:
                    # Fallback: create a temporary conv layer
                    print(
                        f"Warning: No adapter for {C} channels, creating temporary one"
                    )
                    temp_conv = nn.Conv2d(C, self.target_channels, 1).to(
                        spatially_resized.device
                    )
                    channel_adjusted = temp_conv(spatially_resized)

                # Remove batch dimension: 1 c h w -> c h w
                uniform_layers.append(rearrange(channel_adjusted, "1 c h w -> c h w"))

            # Now do DCI grouping
            if uniform_layers:
                # Stack layers: list of [c h w] -> [L c h w]
                stacked = rearrange(uniform_layers, "L c h w -> L c h w")

                L, C, H, W = stacked.shape
                G = self.num_groups
                M = max(1, L // G)

                grouped_features = []
                for g in range(min(G, L)):
                    start_idx = g * M
                    end_idx = min((g + 1) * M, L)
                    if start_idx < L:
                        group_avg = torch.mean(stacked[start_idx:end_idx], dim=0)
                        grouped_features.append(group_avg)

                # Add final layer
                grouped_features.append(stacked[-1])

                # Stack grouped features: list of [c h w] -> [G+1 c h w]
                result = rearrange(grouped_features, "G c h w -> G c h w")
                # Add batch dimension: [G+1 c h w] -> [1 G+1 c h w]
                batch_results.append(rearrange(result, "G c h w -> 1 G c h w"))
            else:
                batch_results.append(
                    torch.zeros(
                        1, self.num_groups + 1, self.target_channels, *target_size
                    )
                )

        return torch.cat(batch_results, dim=0)  # [B G+1 c h w]

    def forward(self, multi_layer_features_batch):
        # Handle the batch of lists
        x = self.dense_channel_integration(multi_layer_features_batch)

        # Reshape for MLP: [B G+1 C H W] -> [B H*W (G+1)*C]
        x = rearrange(x, "B G C H W -> B (H W) (G C)")

        return self.mlp(x)


class MultimodalLlama(nn.Module):
    def __init__(self, vision_dim=2048, llama_dim=2048, num_groups=4):
        super().__init__()

        # Vision components
        self.vision_projection = VisionProjection(vision_dim, llama_dim, num_groups)

        # Load Llama model
        self.llama = AutoModelForCausalLM.from_pretrained(
            "alpindale/Llama-3.2-1B-Instruct"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "alpindale/Llama-3.2-1B-Instruct"
        )

        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<image>", "</image>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # Store image token ID
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def encode_image(self, multi_layer_features):
        # Project multi-layer features to Llama dimension using DCI
        return self.vision_projection(multi_layer_features)

    def forward(
        self, input_ids, multi_layer_features=None, attention_mask=None, labels=None
    ):
        if multi_layer_features is not None:
            with torch.no_grad():
                inputs_embeds = self.llama.get_input_embeddings()(input_ids).detach()

            inputs_embeds = inputs_embeds.clone().detach().requires_grad_(True)
            image_embeds = self.encode_image(multi_layer_features)

            image_token_pos = torch.where(input_ids == self.image_token_id)

            if len(image_token_pos[0]) > 0:
                pos = image_token_pos[1][0]

                # Concatenate embeddings
                inputs_embeds = torch.cat(
                    [inputs_embeds[:, :pos], image_embeds, inputs_embeds[:, pos + 1 :]],
                    dim=1,
                )

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask[:, :pos],
                            torch.ones(
                                (attention_mask.shape[0], image_embeds.shape[1]),
                                device=attention_mask.device,
                            ),
                            attention_mask[:, pos + 1 :],
                        ],
                        dim=1,
                    )

                # Update labels to match new sequence length
                if labels is not None:
                    labels = torch.cat(
                        [
                            labels[:, :pos],
                            torch.full(
                                (labels.shape[0], image_embeds.shape[1]),
                                -100,
                                device=labels.device,
                            ),
                            labels[:, pos + 1 :],
                        ],
                        dim=1,
                    )

            return self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            return self.llama(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

    def generate(self, input_ids, multi_layer_features=None, streamer=None, **kwargs):
        if multi_layer_features is not None:
            inputs_embeds = self.llama.get_input_embeddings()(input_ids)
            image_embeds = self.encode_image(multi_layer_features)

            # Insert image embeddings
            image_token_pos = (input_ids == self.image_token_id).nonzero()
            if len(image_token_pos) > 0:
                pos = image_token_pos[0, 1]
                inputs_embeds = torch.cat(
                    [inputs_embeds[:, :pos], image_embeds, inputs_embeds[:, pos + 1 :]],
                    dim=1,
                )

            return self.llama.generate(
                inputs_embeds=inputs_embeds, streamer=streamer, **kwargs
            )
        else:
            return self.llama.generate(input_ids=input_ids, streamer=streamer, **kwargs)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print("Loading model weights...")

        # Download weights if needed
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Setup ResNet vision encoder
        self.resnet = timm.create_model(
            "resnet101_clip.openai", pretrained=True, features_only=True
        )
        self.resnet.to("cuda")
        self.resnet.eval()

        # Setup preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load multimodal model
        self.model = MultimodalLlama(vision_dim=2048, llama_dim=2048, num_groups=2)

        # Set up for inference only
        self.model.llama.eval()
        for param in self.model.llama.parameters():
            param.requires_grad = False

        for param in self.model.vision_projection.parameters():
            param.requires_grad = False

        # Resize token embeddings
        self.model.llama.resize_token_embeddings(len(self.model.tokenizer))

        # Load trained weights
        model_path = os.path.join(MODEL_CACHE, "model.pt")  # Adjust path as needed
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cuda"))

        self.model.to("cuda")
        self.tokenizer = self.model.tokenizer

        print("Setup took: ", time.time() - start)

    def get_multi_layer_embeddings(self, image_path):
        """Extract multi-layer embeddings from ResNet"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img_tensor: torch.Tensor = self.preprocess(img)  # type: ignore
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to("cuda")

            # Generate multi-layer embeddings
            with torch.no_grad():
                features = self.resnet(img_tensor)

                # Convert to list of tensors
                feature_list = []
                shapes = []

                for feat in features:
                    feat_tensor = feat.squeeze(0)  # Remove batch dim
                    feature_list.append(feat_tensor)
                    shapes.append(feat_tensor.shape)

                return feature_list, shapes
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe this image"),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=512,
            ge=1,
            le=2048,
        ),
        temperature: float = Input(
            description="Temperature for sampling", default=0.7, ge=0.1, le=2.0
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling parameter",
            default=0.6,
            ge=0.0,
            le=1.0,
        ),
        top_k: int = Input(
            description="Top-k sampling parameter (0 to disable)",
            default=0,
            ge=0,
            le=100,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty", default=1.0, ge=0.1, le=2.0
        ),
        do_sample: bool = Input(description="Whether to use sampling", default=True),
        num_beams: int = Input(
            description="Number of beams for beam search (1 for no beam search)",
            default=1,
            ge=1,
            le=8,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        # Get image embeddings
        multi_layer_features, shapes = self.get_multi_layer_embeddings(image)
        if multi_layer_features is None:
            yield "Error processing image"
            return

        # Convert numpy arrays to tensors if needed
        feature_tensors = []
        for feat in multi_layer_features:
            if isinstance(feat, np.ndarray):
                feat_tensor = torch.tensor(feat).float().to("cuda")
            else:
                feat_tensor = feat.to("cuda")
            feature_tensors.append(feat_tensor)

        # Prepare the prompt
        full_prompt = f"<image>\n{prompt} "

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens + 100,  # Add buffer for prompt
        ).to("cuda")

        # Setup generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_beams": num_beams,
            "use_cache": False,
        }

        # Add top_k if specified
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

        # Setup streaming
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer

        # Start generation in separate thread
        thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": inputs.input_ids,
                "multi_layer_features": [feature_tensors],  # Wrap in list for batch
                **generation_kwargs,
            },
        )
        thread.start()

        # Stream the response
        full_response = ""
        for new_text in streamer:
            # Clean the text and remove the original prompt
            clean_text = re.sub("<$|<END$", "", new_text)
            if full_prompt in clean_text:
                clean_text = clean_text.replace(full_prompt, "")

            # Only yield new parts
            if len(clean_text) > len(full_response):
                new_part = clean_text[len(full_response) :]
                full_response = clean_text
                yield new_part

        thread.join()
