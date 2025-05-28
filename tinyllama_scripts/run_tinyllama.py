import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

def main():
    print("Loading TinyLlama model...")
    model_path = "/home/rubel/gem5/tinyllama_model"

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Inference result:")
    print(result)

if __name__ == "__main__":
    main()

