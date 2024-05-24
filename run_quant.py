from huggingface_hub import login
login("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")

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

from corpora import conversation_data
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = GPTQConfig(bits=4, dataset = conversation_data[:600], tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

model.push_to_hub("Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ-Phil") # change your hf username