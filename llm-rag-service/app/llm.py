from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class LLMService:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            #device_map="auto"
        )
        
        # Create pipeline for easier inference
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate text based on the input prompt using the Mistral model.
        
        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity of generation via nucleus sampling
            **kwargs: Additional arguments passed to the pipeline
            
        Returns:
            Generated text as string
        """
        # Format prompt according to Mistral's instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate text
        result = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs
        )[0]["generated_text"]
        
        # Remove the instruction prompt from the output
        response = result.split("[/INST]")[-1].strip()
        return response

# Create a global instance
llm_service = LLMService()
