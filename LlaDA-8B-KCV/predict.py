from cog import BasePredictor, Input, ConcatenateIterator, Path, BaseModel
from transformers import AutoTokenizer, AutoModel
from peft.peft_model import PeftModel
from typing import Union, List
from threading import Thread

import torch.nn.functional as F
import numpy as np
import torch

PROMPT_TEMPLATE = """<|startoftext|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
SYSTEM_PROMPT = "Asisten adalah chatbot bernama Anita, seorang chatbot yang ramah."


def load_llada_model(
    base_model_name="GSAI-ML/LLaDA-8B-Instruct",
    adapter_path="./lora-llada",
    device="cuda",
):
    # Load base model
    model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_with_llada_states(
    model,
    tokenizer,
    prompt,
    gen_length=64,
    steps=32,
    temperature=0.5,
    cfg_scale=0.0,
    block_length=32,
    remasking="low_confidence",
    constraints=None,
    device="cuda",
):
    """
    Modified version that returns all intermediate states.
    Returns: List of (sequence, confidence_scores) tuples for each step
    """
    MASK_ID = 126336
    MASK_SYMBOL = "[MASK]"
    states = []

    # Process constraints
    if constraints is None:
        constraints = {}

    def decode_with_masks(tensor):
        text = []
        for token_id in tensor[0]:  # Take first batch
            if token_id == MASK_ID:
                text.append(MASK_SYMBOL)
            else:
                # Decode single token while preserving whitespace
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                text.append(token_text)
        return "".join(text)

    # Convert string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id

    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    # Initialize sequence with masks
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(
        device
    )
    x[:, :prompt_length] = input_ids.clone()

    # Save initial state
    states.append(decode_with_masks(x))

    # Apply initial constraints
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id

    prompt_index = x != MASK_ID
    block_length = min(block_length, gen_length)
    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(1, steps // num_blocks)

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])

        block_mask_index = x[:, block_start:block_end] == MASK_ID
        if not block_mask_index.any():
            continue

        mask_count = block_mask_index.sum(dim=1, keepdim=True)
        base_tokens = mask_count // steps_per_block
        remainder = mask_count % steps_per_block
        num_transfer_tokens = (
            torch.zeros(
                mask_count.size(0), steps_per_block, device=device, dtype=torch.int64
            )
            + base_tokens
        )
        num_transfer_tokens[:, :remainder] += 1

        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            if not mask_index.any():
                break

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if temperature > 0:
                logits = logits.to(torch.float64)
                noise = torch.rand_like(logits, dtype=torch.float64)
                gumbel_noise = (-torch.log(noise)) ** temperature
                logits = logits.exp() / gumbel_noise

            x0 = torch.argmax(logits, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=device)

            x0_p[:, block_end:] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(float("-inf")).to(device)
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:
                    _, indices = torch.topk(
                        block_confidence,
                        k=min(
                            num_transfer_tokens[j, i].item(), block_confidence.numel()
                        ),  # type: ignore
                    )
                    transfer_index[j, indices + block_start] = True
                else:
                    transfer_index[j, block_start:block_end] = mask_index[
                        j, block_start:block_end
                    ]

            x = torch.where(transfer_index, x0, x)

            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id

            states.append(decode_with_masks(x))

    return states


class Output(BaseModel):
    """
    Output class for the LLaDA predictor, containing generated text states
    """

    states: List[str]
    final_output: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model, self.tokenizer = load_llada_model()

    def predict(
        self,
        prompt: str = Input(description="Input text for generation"),
        system_prompt: str = Input(
            description="System prompt to guide model behavior",
            default=SYSTEM_PROMPT,
        ),
        prompt_template: str = Input(
            description="Template for formatting prompts",
            default=PROMPT_TEMPLATE,
        ),
        max_tokens: int = Input(
            description="Maximum length of generated text", default=256, ge=1
        ),
        temperature: float = Input(
            description="Sampling temperature (higher = more random)",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        steps: int = Input(
            description="Number of denoising steps per block", default=32, ge=1
        ),
        block_length: int = Input(
            description="Length of parallel generation blocks", default=32, ge=1
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale", default=0.0, ge=0.0
        ),
        seed: int = Input(
            description="Random seed for reproducible generation", default=None
        ),
    ) -> Output:
        """Run generation with LLaDA model"""

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # Format full prompt
        full_prompt = prompt_template.format(prompt=prompt, system_prompt=system_prompt)

        # Generate states
        states = generate_with_llada_states(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            gen_length=max_tokens,
            temperature=temperature,
            steps=steps,
            block_length=block_length,
            cfg_scale=cfg_scale,
            device="cuda",
        )

        # Get final output (last state)
        final_output = states[-1] if states else ""

        # Return Output object with states and final output
        return Output(states=states, final_output=final_output)
