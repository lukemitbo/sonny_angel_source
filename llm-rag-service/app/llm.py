from typing import Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .rag import Retriever


class LLMService:

    def __init__(self, retriever: Optional[Retriever] = None):
        self.model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.retriever = retriever

    def _initialize(self):
        if self.pipe is not None:
            return

        try:
            # Load model and tokenizer
            print(f"Loading model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
                # device_map can be enabled when GPU is present
                # device_map="auto"
            )
            print("Model loaded successfully!")
            # Create pipeline for easier inference
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                #trust_remote_code=True,
            )
            print("Pipeline loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **kwargs: Dict[str, Any]) -> str:
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
        # Ensure model is initialized
        self._initialize()

        # Format prompt according to Mistral's instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Generate text
        result = self.pipe(formatted_prompt,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature,
                           top_p=top_p,
                           do_sample=True,
                           **kwargs)[0]["generated_text"]

        # Remove the instruction prompt from the output
        response = result.split("[/INST]")[-1].strip()
        return response

    def query_with_rag(self,
                       query: str,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       k: int = 3,
                       **kwargs: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: The user's question
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Controls diversity of generation
            k: Number of relevant passages to retrieve
            **kwargs: Additional arguments passed to generate
            
        Returns:
            Tuple of (context used, generated response)
        """
        # Retrieve relevant context if retriever available
        if self.retriever is None:
            response = self.generate(query,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature,
                                     top_p=top_p,
                                     **kwargs)
            return "", response

        context_results = self.retriever.retrieve(query, k=k)
        context = "\n\n".join(text for text, _ in context_results)

        # Build prompt with context
        prompt = f"""Use only the following context to answer the question. If the context doesn't contain relevant information, say so and refuse to answer.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response with context
        response = self.generate(prompt,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature,
                                 top_p=top_p,
                                 **kwargs)

        return context, response
