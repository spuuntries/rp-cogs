import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from repeng import ControlVector, ControlModel, DatasetEntry
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
import numpy as np
import json
import re
import os
import tempfile
import shutil # For cleaning up temporary directory

# Define the constants for chat templating
system_tag, user_tag, asst_tag, eos_tag = "<|im_start|>system\n", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model and tokenizer into memory."""
        self.model_name = "NousResearch/Hermes-2-Pro-Mistral-7B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./model"
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            "./model",
            torch_dtype=torch.float16,
            device_map="auto" 
        )

        # Load suffixes from true_facts.json. 
        try:
            with open("true_facts.json", "r") as f:
                self.suffixes = json.load(f)
        except FileNotFoundError:
            print("true_facts.json not found. Ensure it's copied during build. Using dummy data for demonstration.")
            self.suffixes = [
                "The sky is blue.",
                "Water is wet.",
                "The sun is a star.",
                "Birds can fly.",
                "Fish swim in water."
            ]


    def _generate_antonym(self, attribute: str) -> str:
        """
        Uses the base model to generate a single-word antonym for the given attribute.
        """
        self.base_model.eval() # Ensure the model is in evaluation mode

        prompt = f"{user_tag}What is the single-word opposite of '{attribute}'? REPLY WITH JUST THE WORD.{eos_tag}{asst_tag}The opposite is '"
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        with torch.no_grad(): # Disable gradient calculations for inference
            output = self.base_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.1,
                repetition_penalty=1.2,
                attention_mask=attention_mask,
            )

        full_decoded_text = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response_text = full_decoded_text.strip()
        antonym = response_text.split()[0].strip().replace('.', '').replace(',', '').replace("'", "")
        return antonym

    def _generate_examples(self, persona: str, num_examples: int) -> list[str]:
        """
        Generates a list of short, descriptive scenarios for a given persona using the base model.
        """
        self.base_model.eval()

        prompt = (
            f"{user_tag}I need short, one-sentence descriptions of a person's behavior if the person is {persona}. "
            f"Provide a numbered list of {num_examples} distinct examples. YOU MUST PROVIDE {num_examples} items. For example: 1. a compassionate friend. 2. a sympathetic listener."
            f"Do not write anything else, just the numbered list. {eos_tag}{asst_tag}"
        )
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        with torch.no_grad():
            output = self.base_model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.6,
                do_sample=True,
                attention_mask=attention_mask
            )

        full_decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response part
        response_start_index = full_decoded_text.find(asst_tag)
        if response_start_index != -1:
            response_text = full_decoded_text[response_start_index + len(asst_tag):].strip()
        else:
            response_text = full_decoded_text.strip() # Fallback

        # Use regex to find all numbered list items (e.g., "1. ...", "2. ...")
        examples = re.findall(r'^\s*\d+\.\s*(.*)', response_text, re.MULTILINE)
        return examples[:num_examples] 

    def _make_dataset(
        self,
        template: str,
        positive_personas: list[str],
        negative_personas: list[str],
    ) -> list[DatasetEntry]:
        """
        Creates a list of DatasetEntry objects for training the ControlVector.
        """
        dataset = []
        # Pair positive and negative personas for dataset entries.
        min_len = min(len(positive_personas), len(negative_personas))

        for suffix in self.suffixes: 
            for i in range(min_len):
                positive_persona = positive_personas[i]
                negative_persona = negative_personas[i]

                positive_template = template.format(persona=positive_persona)
                negative_template = template.format(persona=negative_persona)
                dataset.append(
                    DatasetEntry(
                        positive=f"{system_tag}{positive_template}{eos_tag}{user_tag}Hey{eos_tag}{asst_tag}{suffix}",
                        negative=f"{system_tag}{negative_template}{eos_tag}{user_tag}Hey{eos_tag}{asst_tag}{suffix}",
                    )
                )
        return dataset

    def predict(
        self,
        attributes_to_generate: str = Input(
            description="Comma-separated list of attributes for which to generate control vectors (e.g., 'girly,modestly,verbose,happy')",
            default="girly,modestly,verbose,happy"
        ),
        num_examples_per_side: int = Input(
            description="Number of descriptive examples to generate for each side of the contrast. "
                        "More examples might lead to better vectors but will increase generation time.",
            default=3,
            ge=1,
            le=10 # Cap the number of examples to prevent excessively long runtimes
        ),
        seed: int = Input(
            description="Seed for reproducibility of example generation and vector training. Set to 0 for random behavior.",
            default=None
        )
    ) -> list[Path]:
        """
        Generates control vectors for a list of specified attributes.
        Returns a list of Path objects, each pointing to a generated .gguf file.
        """
        if seed is not None:
            set_seed(seed)
            print(f"Set seed to: {seed}")

        # Parse the input attributes string into a list
        attributes = [attr.strip() for attr in attributes_to_generate.split(',') if attr.strip()]
        if not attributes:
            raise ValueError("No attributes provided. Please provide a comma-separated list of attributes.")

        output_paths = []
        # Create a temporary directory to store the generated GGUF files.
        # Cog will automatically copy these files when Path objects are returned.
        tmpdir = tempfile.mkdtemp()
        try:
            for attribute in attributes:
                print(f"--- Processing attribute: '{attribute}' ---")

                # Generate the contrastive attribute (antonym)
                print(f"Generating antonym for '{attribute}'...")
                positive_persona = attribute
                negative_persona = self._generate_antonym(positive_persona)
                print(f"Generated contrastive pair: ('{positive_persona}', '{negative_persona}')")

                # Generate descriptive examples for both positive and negative personas
                print(f"Generating {num_examples_per_side} examples for each side...")
                positive_examples = self._generate_examples(f"a very {positive_persona} person", num_examples_per_side) + [f"a very {positive_persona} person"]
                negative_examples = self._generate_examples(f"a very {negative_persona} person", num_examples_per_side) + [f"a very {negative_persona} person"]

                # Ensure sufficient examples were generated
                expected_total_examples = num_examples_per_side + 1 # +1 for the appended direct persona
                if len(positive_examples) < expected_total_examples or len(negative_examples) < expected_total_examples:
                    print(f"Warning: Could not generate enough examples for '{attribute}'. Expected {expected_total_examples} per side, but got {len(positive_examples)} and {len(negative_examples)}. Skipping vector generation for this attribute.")
                    continue

                print("\nPositive Examples:")
                for ex in positive_examples: print(f"- {ex}")
                print("\nNegative Examples:")
                for ex in negative_examples: print(f"- {ex}")

                # Create the dataset for ControlVector training
                dataset = self._make_dataset(
                    "Pretend you're a person who acts like this: '{persona}'. Now, make a statement about the world.",
                    positive_examples,
                    negative_examples,
                )
                print(f"\nCreated dataset with {len(dataset)} entries.")

                # Wrap the base model with ControlModel and train the control vector
                print("Training control vector...")
                control_model_wrapper = ControlModel(self.base_model, list(range(-5, -18, -1)))
                control_model_wrapper.reset() # Reset the control model's state

                control_vector = ControlVector.train(
                    control_model_wrapper,
                    self.tokenizer,
                    dataset,
                    method="pca_center" # TODO: Tentative tbh
                )
                control_model_wrapper.reset() # Reset again after training
                # Unwrap the model to ensure self.base_model is returned to its original state
                self.base_model = control_model_wrapper.unwrap()

                # Save the generated control vector as a GGUF file in the temporary directory
                output_filename = os.path.join(tmpdir, f"{attribute}.gguf")
                control_vector.export_gguf(output_filename)
                output_paths.append(Path(output_filename)) # Add the Path object to the list
                print(f"Successfully generated vector for '{attribute}' and saved to {output_filename}.\n")
        finally:
            # Clean up the temporary directory after all files have been processed
            shutil.rmtree(tmpdir)

        return output_paths
