import sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/tokenizer.json")
tokens = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 198, 248046, 198, 248045, 846, 198, 12195, 369, 279, 6511, 314, 37973, 248046, 198, 248045, 74455, 198, 248068, 198]
sys.stdout.write("TOK: " + str([tokenizer.decode([t]) for t in tokens]) + "\n")
