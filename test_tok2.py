import sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/tokenizer.json")
sys.stdout.write("USER TURN: " + str(tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nwhat is the capital of Lebanon<|im_end|>\n<|im_start|>assistant\n<think>\n").ids) + "\n")
