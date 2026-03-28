import sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/tokenizer.json")
tokens = [248045, 846, 198, 814, 20139, 20340, 12, 1020, 12, 4431, 15089, 303, 799, 13901, 13, 248046, 198, 248045, 74455, 198]
sys.stdout.write("USER TURN: " + str([tokenizer.decode([t]) for t in tokens]) + "\n")
tokens_sys = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593, 26003, 248046, 198]
sys.stdout.write("SYS TURN: " + str([tokenizer.decode([t]) for t in tokens_sys]) + "\n")
