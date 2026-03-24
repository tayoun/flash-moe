#!/usr/bin/env python3
"""Build CAR popularity table from co-occurrence data.

For each (layer, expert), finds the expert most frequently co-occurring with it.
This is the best substitute candidate — when expert E is not cached,
the most frequent co-occurring expert is likely to produce similar outputs.

Output: JSON file with {layer_N: {expert_id: best_substitute_id, ...}}
"""
import json
import sys

def main():
    cooccur_path = sys.argv[1] if len(sys.argv) > 1 else "cooccur_122b.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "car_table_122b.json"

    with open(cooccur_path) as f:
        data = json.load(f)

    num_layers = data["num_layers"]
    num_experts = data["num_experts"]
    result = {}

    for layer_key, matrix in data["layers"].items():
        layer_id = int(layer_key)
        table = {}
        for expert_id in range(num_experts):
            row = matrix[expert_id]
            # Find expert with highest co-occurrence (excluding self)
            best_sub = -1
            best_count = 0
            for e in range(num_experts):
                if e == expert_id:
                    continue
                if row[e] > best_count:
                    best_count = row[e]
                    best_sub = e
            if best_sub >= 0 and best_count > 0:
                table[expert_id] = best_sub
        result[f"layer_{layer_id}"] = table

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"Built CAR table: {num_layers} layers, {num_experts} experts -> {output_path}")

if __name__ == "__main__":
    main()
