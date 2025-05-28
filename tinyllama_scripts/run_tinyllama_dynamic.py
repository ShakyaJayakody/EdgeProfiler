import sys
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Prompt: {prompt}")
    print(f"Generating up to {max_tokens} tokens...")

    model_path = "/home/rubel/gem5/tinyllama_model"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Response:")
    print(result)

if __name__ == "__main__":
    main()

