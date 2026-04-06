#!/usr/bin/env python3
"""Validate Gemma expert index metadata before packing."""

import argparse
import json
import sys


BF16_BYTES = 2


def expect_shape(component, num_experts, hidden, moe_intermediate):
    if component == "gate_up_proj":
        return [num_experts, 2 * moe_intermediate, hidden]
    if component == "down_proj":
        return [num_experts, hidden, moe_intermediate]
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate Gemma expert metadata-only index")
    parser.add_argument("--index", required=True, help="Path to Gemma expert index JSON")
    parser.add_argument("--allow-missing-layers", action="store_true",
                        help="Do not fail if some layers are missing")
    args = parser.parse_args()

    with open(args.index) as f:
        payload = json.load(f)

    errors = []
    warnings = []

    layout = payload.get("layout", {})
    components = layout.get("components", [])
    comp_by_name = {c["name"]: c for c in components}
    expert_reads = payload.get("expert_reads", {})
    model_cfg = payload.get("model_config", {})

    num_layers = int(layout.get("num_layers", model_cfg.get("num_hidden_layers", len(expert_reads))))
    num_experts = int(layout.get("num_experts", model_cfg.get("num_experts", 0)))
    hidden = int(model_cfg.get("hidden_size", 0))
    moe_intermediate = int(model_cfg.get("moe_intermediate_size", 0))

    required_components = ["gate_up_proj", "down_proj"]
    for comp in required_components:
        if comp not in comp_by_name:
            errors.append(f"layout missing required component: {comp}")

    if num_experts <= 0:
        errors.append("num_experts must be > 0")
    if num_layers <= 0:
        errors.append("num_layers must be > 0")

    if not args.allow_missing_layers and len(expert_reads) != num_layers:
        errors.append(f"layer count mismatch: index has {len(expert_reads)}, expected {num_layers}")

    per_component_shapes = {name: set() for name in required_components}
    per_component_sizes = {name: set() for name in required_components}
    per_component_strides = {name: set() for name in required_components}
    per_component_dtypes = {name: set() for name in required_components}

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_info = expert_reads.get(layer_key)
        if layer_info is None:
            if not args.allow_missing_layers:
                errors.append(f"missing layer {layer_idx}")
            continue

        for comp in required_components:
            info = layer_info.get(comp)
            if info is None:
                errors.append(f"layer {layer_idx}: missing component {comp}")
                continue

            dtype = info.get("dtype")
            shape = info.get("shape")
            expert_size = int(info.get("expert_size", -1))
            stride = int(info.get("expert_stride", -1))
            total_size = int(info.get("total_size", -1))

            per_component_dtypes[comp].add(dtype)
            per_component_sizes[comp].add(expert_size)
            per_component_strides[comp].add(stride)
            if isinstance(shape, list):
                per_component_shapes[comp].add(tuple(shape))

            if dtype != "BF16":
                errors.append(f"layer {layer_idx} {comp}: dtype={dtype}, expected BF16")

            if not isinstance(shape, list) or len(shape) != 3:
                errors.append(f"layer {layer_idx} {comp}: invalid shape {shape}")
            else:
                expected = expect_shape(comp, num_experts, hidden, moe_intermediate)
                if expected and shape != expected:
                    errors.append(
                        f"layer {layer_idx} {comp}: shape={shape}, expected={expected}"
                    )

            if expert_size <= 0:
                errors.append(f"layer {layer_idx} {comp}: invalid expert_size={expert_size}")
            else:
                if stride != expert_size:
                    errors.append(
                        f"layer {layer_idx} {comp}: expert_stride={stride} expected contiguous stride={expert_size}"
                    )

            if total_size != num_experts * expert_size:
                errors.append(
                    f"layer {layer_idx} {comp}: total_size={total_size}, expected={num_experts * expert_size}"
                )

            if isinstance(shape, list) and len(shape) == 3 and dtype == "BF16":
                expected_size = shape[1] * shape[2] * BF16_BYTES
                if expert_size != expected_size:
                    errors.append(
                        f"layer {layer_idx} {comp}: expert_size={expert_size}, expected from shape={expected_size}"
                    )

    for comp in required_components:
        if len(per_component_dtypes[comp]) > 1:
            errors.append(f"{comp}: inconsistent dtypes across layers: {sorted(per_component_dtypes[comp])}")
        if len(per_component_shapes[comp]) > 1:
            errors.append(f"{comp}: inconsistent shapes across layers: {sorted(per_component_shapes[comp])}")
        if len(per_component_sizes[comp]) > 1:
            errors.append(f"{comp}: inconsistent expert_size across layers: {sorted(per_component_sizes[comp])}")
        if len(per_component_strides[comp]) > 1:
            errors.append(f"{comp}: inconsistent expert_stride across layers: {sorted(per_component_strides[comp])}")

    layout_expert_size = int(layout.get("expert_size", 0))
    gate_size = int(comp_by_name.get("gate_up_proj", {}).get("size", 0))
    down_size = int(comp_by_name.get("down_proj", {}).get("size", 0))
    if layout_expert_size != gate_size + down_size:
        errors.append(
            f"layout expert_size={layout_expert_size} but components sum to {gate_size + down_size}"
        )

    missing = payload.get("missing_tensors", [])
    if missing:
        warnings.append(f"index lists {len(missing)} missing_tensors")

    print(f"Index: {args.index}")
    print(f"Layers expected: {num_layers}, indexed: {len(expert_reads)}")
    print(f"Experts/layer: {num_experts}")
    print(f"Layout architecture: {layout.get('architecture', 'unknown')}")
    for w in warnings:
        print(f"WARNING: {w}")

    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Validation PASSED")


if __name__ == "__main__":
    main()
