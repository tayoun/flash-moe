#!/usr/bin/env python3
"""
Trace-driven cache simulator for Flash-MoE expert access patterns.

Reads binary trace files produced by `./infer --trace <path>` and simulates
different cache policies and sizes to estimate performance without running
the full inference pipeline.

Trace format:
  Header: int32 num_layers, int32 num_experts, int32 K, int32 expert_size
  Per-token: int32[K] expert_indices × num_layers

Usage:
  python simulate.py trace.bin --cache-mb 4096 --policy arc
  python simulate.py trace.bin --sweep  # test multiple configurations
"""

import argparse
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class TraceHeader:
    num_layers: int
    num_experts: int
    K: int
    expert_size: int


def read_trace(path: str) -> Tuple[TraceHeader, List[List[List[int]]]]:
    """Read trace file and return header + list of tokens, each containing layer expert indices."""
    MAX_K = 8  # MAX_K from infer.m - we always write 8 int32s per layer

    with open(path, 'rb') as f:
        # Read header
        hdr_data = f.read(16)
        if len(hdr_data) < 16:
            raise ValueError("Trace file too small for header")
        num_layers, num_experts, K, expert_size = struct.unpack('iiii', hdr_data)
        header = TraceHeader(num_layers, num_experts, K, expert_size)

        # Read per-token routing data
        # Note: infer.m writes MAX_K (8) int32s per layer, padded with -1
        tokens = []
        record_size = MAX_K * 4 * num_layers  # MAX_K int32s per layer

        while True:
            data = f.read(record_size)
            if len(data) < record_size:
                break

            # Parse token's routing: num_layers × MAX_K expert indices
            token_routing = []
            offset = 0
            for _ in range(num_layers):
                layer_experts = list(struct.unpack(f'{MAX_K}i', data[offset:offset + MAX_K*4]))
                # Filter out -1 (padding for K < MAX_K)
                layer_experts = [e for e in layer_experts if e >= 0]
                token_routing.append(layer_experts)
                offset += MAX_K * 4
            tokens.append(token_routing)

        return header, tokens


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[Tuple[int, int], bool] = OrderedDict()

    def access(self, layer: int, expert: int) -> bool:
        """Access (layer, expert). Returns True if hit, False if miss."""
        key = (layer, expert)
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = True
            return False


class ARCCache:
    """Adaptive Replacement Cache implementation."""

    def __init__(self, capacity: int):
        self.c = capacity
        self.p = 0  # target size of T1

        # Recent items
        self.t1: OrderedDict[Tuple[int, int], bool] = OrderedDict()  # recent, seen once
        self.t2: OrderedDict[Tuple[int, int], bool] = OrderedDict()  # frequent, seen >= twice

        # Ghost lists (evicted items)
        self.b1: OrderedDict[Tuple[int, int], bool] = OrderedDict()  # evicted from t1
        self.b2: OrderedDict[Tuple[int, int], bool] = OrderedDict()  # evicted from t2

    def access(self, layer: int, expert: int) -> bool:
        """Access (layer, expert). Returns True if hit, False if miss."""
        key = (layer, expert)

        # Case 1: hit in T1 or T2
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = True
            return True

        if key in self.t2:
            self.t2.move_to_end(key)
            return True

        # Miss - check ghost lists to adapt
        if key in self.b1:
            # Increase target size of recent list
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.p + delta, self.c)
            self._replace(key, in_b2=False)
            del self.b1[key]
            self.t2[key] = True
            return False

        if key in self.b2:
            # Decrease target size of recent list
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(self.p - delta, 0)
            self._replace(key, in_b2=True)
            del self.b2[key]
            self.t2[key] = True
            return False

        # Complete miss - add to T1
        total_t = len(self.t1) + len(self.t2)
        total_b = len(self.b1) + len(self.b2)

        if total_t == self.c:
            if len(self.t1) < self.c:
                if len(self.b1) > 0:
                    self.b1.popitem(last=False)
                self._replace(key, in_b2=False)
            else:
                # T1 is full, discard from T1
                evicted = self.t1.popitem(last=False)
        elif total_t < self.c and total_t + total_b >= self.c:
            if total_t + total_b >= 2 * self.c:
                if len(self.b2) > 0:
                    self.b2.popitem(last=False)
            self._replace(key, in_b2=False)

        self.t1[key] = True
        return False

    def _replace(self, key: Tuple[int, int], in_b2: bool):
        """Replace an item from cache."""
        if len(self.t1) > 0 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            evicted = self.t1.popitem(last=False)
            self.b1[evicted[0]] = True
            # Limit ghost list size
            if len(self.b1) > self.c:
                self.b1.popitem(last=False)
        elif len(self.t2) > 0:
            evicted = self.t2.popitem(last=False)
            self.b2[evicted[0]] = True
            if len(self.b2) > self.c:
                self.b2.popitem(last=False)


def simulate(header: TraceHeader, tokens: List[List[List[int]]],
             cache_mb: float, policy: str) -> Dict:
    """
    Simulate cache behavior over the trace.

    Returns dict with:
      - hit_rate: fraction of expert accesses that hit cache
      - total_accesses: total expert reads
      - total_misses: cache misses
      - estimated_io_ms: estimated I/O time per token (based on miss rate)
      - estimated_tok_s: estimated tokens/second
    """
    # Calculate cache capacity in experts
    cache_bytes = int(cache_mb * 1024 * 1024)
    cache_capacity = cache_bytes // header.expert_size

    if cache_capacity == 0:
        cache_capacity = 1  # Minimum 1 expert

    # Initialize cache
    if policy == 'lru':
        cache = LRUCache(cache_capacity)
    elif policy == 'arc':
        cache = ARCCache(cache_capacity)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Simulate
    total_accesses = 0
    total_hits = 0

    per_token_misses = []

    for token_idx, token_routing in enumerate(tokens):
        token_misses = 0
        for layer_idx, experts in enumerate(token_routing):
            for expert in experts:
                total_accesses += 1
                if cache.access(layer_idx, expert):
                    total_hits += 1
                else:
                    token_misses += 1
        per_token_misses.append(token_misses)

    total_misses = total_accesses - total_hits
    hit_rate = total_hits / total_accesses if total_accesses > 0 else 0

    # Estimate I/O time per token
    # Assumptions for M4 Mac mini SSD:
    #   - Cold read: ~5.5 GB/s (parallel pread)
    #   - Expert size from header
    avg_misses_per_token = total_misses / len(tokens) if tokens else 0
    io_bytes_per_token = avg_misses_per_token * header.expert_size
    ssd_bandwidth_gbs = 5.5
    io_ms_per_token = (io_bytes_per_token / (ssd_bandwidth_gbs * 1e9)) * 1000

    # Estimate GPU compute time per token (relatively fixed)
    # Based on observed 122B: ~200ms total, ~74% I/O bound at baseline
    # So GPU compute ≈ 52ms
    gpu_ms_per_token = 52.0

    total_ms_per_token = gpu_ms_per_token + io_ms_per_token
    estimated_tok_s = 1000.0 / total_ms_per_token if total_ms_per_token > 0 else 0

    return {
        'cache_mb': cache_mb,
        'cache_capacity': cache_capacity,
        'policy': policy,
        'total_tokens': len(tokens),
        'total_accesses': total_accesses,
        'total_hits': total_hits,
        'total_misses': total_misses,
        'hit_rate': hit_rate,
        'avg_misses_per_token': avg_misses_per_token,
        'io_bytes_per_token': io_bytes_per_token,
        'io_ms_per_token': io_ms_per_token,
        'gpu_ms_per_token': gpu_ms_per_token,
        'total_ms_per_token': total_ms_per_token,
        'estimated_tok_s': estimated_tok_s,
    }


def print_results(results: Dict):
    """Pretty print simulation results."""
    print(f"\n{'='*60}")
    print(f"Cache: {results['cache_mb']:.0f} MB ({results['cache_capacity']} experts), Policy: {results['policy'].upper()}")
    print(f"{'='*60}")
    print(f"Tokens simulated:    {results['total_tokens']}")
    print(f"Total accesses:      {results['total_accesses']}")
    print(f"Cache hits:          {results['total_hits']} ({results['hit_rate']*100:.1f}%)")
    print(f"Cache misses:        {results['total_misses']} ({(1-results['hit_rate'])*100:.1f}%)")
    print(f"Avg misses/token:    {results['avg_misses_per_token']:.1f}")
    print(f"I/O bytes/token:     {results['io_bytes_per_token']/1e6:.1f} MB")
    print(f"Est. I/O ms/token:   {results['io_ms_per_token']:.1f} ms")
    print(f"Est. GPU ms/token:   {results['gpu_ms_per_token']:.1f} ms")
    print(f"Est. total ms/token: {results['total_ms_per_token']:.1f} ms")
    print(f"Est. tok/s:          {results['estimated_tok_s']:.2f}")


def sweep(header: TraceHeader, tokens: List[List[List[int]]]):
    """Run parameter sweep over cache sizes and policies."""
    cache_sizes_mb = [0, 1024, 2048, 4096, 6144, 8192]  # MB
    policies = ['lru', 'arc']

    print(f"\n{'Cache MB':>10} {'Policy':>6} {'Capacity':>10} {'Hit Rate':>10} {'Miss/Tok':>10} {'I/O ms':>10} {'Est tok/s':>12}")
    print("-" * 80)

    all_results = []
    for cache_mb in cache_sizes_mb:
        for policy in policies:
            if cache_mb == 0 and policy == 'arc':
                continue  # Skip 0 MB for ARC (same as LRU)

            results = simulate(header, tokens, cache_mb, policy)
            all_results.append(results)

            print(f"{results['cache_mb']:>10.0f} {results['policy']:>6} "
                  f"{results['cache_capacity']:>10} {results['hit_rate']*100:>9.1f}% "
                  f"{results['avg_misses_per_token']:>10.1f} {results['io_ms_per_token']:>10.1f} "
                  f"{results['estimated_tok_s']:>12.2f}")

    return all_results


def analyze_routing_patterns(header: TraceHeader, tokens: List[List[List[int]]]):
    """Analyze expert routing patterns for insights."""
    print(f"\n{'='*60}")
    print("Routing Pattern Analysis")
    print(f"{'='*60}")

    # Expert frequency per layer
    freq = {}  # (layer, expert) -> count
    for token_routing in tokens:
        for layer_idx, experts in enumerate(token_routing):
            for expert in experts:
                key = (layer_idx, expert)
                freq[key] = freq.get(key, 0) + 1

    # Calculate statistics
    total_tokens = len(tokens)

    # Find hot experts (>10% of tokens)
    hot_threshold = total_tokens * 0.1
    hot_experts = [(k, v) for k, v in freq.items() if v >= hot_threshold]
    hot_experts.sort(key=lambda x: -x[1])

    print(f"Total tokens: {total_tokens}")
    print(f"Total unique (layer, expert) pairs: {len(freq)}")
    print(f"Hot experts (>10% of tokens): {len(hot_experts)}")

    if hot_experts[:10]:
        print("\nTop 10 hot experts:")
        for (layer, expert), count in hot_experts[:10]:
            print(f"  Layer {layer:2d}, Expert {expert:3d}: {count:5d} tokens ({count/total_tokens*100:.1f}%)")

    # Per-layer statistics
    print("\nPer-layer unique experts accessed:")
    for layer in range(header.num_layers):
        layer_experts = set()
        for token_routing in tokens:
            for expert in token_routing[layer]:
                layer_experts.add(expert)
        print(f"  Layer {layer:2d}: {len(layer_experts):3d} / {header.num_experts} experts ({len(layer_experts)/header.num_experts*100:.1f}%)")

    # Co-occurrence analysis (which experts fire together)
    print("\nExpert co-occurrence (same token, same layer):")
    cooccur = {}  # (layer, e1, e2) -> count
    for token_routing in tokens:
        for layer_idx, experts in enumerate(token_routing):
            for i, e1 in enumerate(experts):
                for e2 in experts[i+1:]:
                    key = (layer_idx, min(e1, e2), max(e1, e2))
                    cooccur[key] = cooccur.get(key, 0) + 1

    top_cooccur = sorted(cooccur.items(), key=lambda x: -x[1])[:10]
    for (layer, e1, e2), count in top_cooccur:
        print(f"  Layer {layer:2d}: experts {e1:3d} + {e2:3d} co-occur {count:4d} times")


def main():
    parser = argparse.ArgumentParser(description='Flash-MoE trace-driven cache simulator')
    parser.add_argument('trace', help='Path to trace file from ./infer --trace')
    parser.add_argument('--cache-mb', type=float, default=4096,
                        help='Cache size in MB (default: 4096)')
    parser.add_argument('--policy', choices=['lru', 'arc'], default='arc',
                        help='Cache policy (default: arc)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep over cache sizes and policies')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze routing patterns')

    args = parser.parse_args()

    # Read trace
    print(f"Reading trace: {args.trace}")
    header, tokens = read_trace(args.trace)
    print(f"Header: {header.num_layers} layers, {header.num_experts} experts, K={header.K}, expert_size={header.expert_size/1e6:.2f} MB")
    print(f"Loaded: {len(tokens)} tokens")

    if args.analyze:
        analyze_routing_patterns(header, tokens)

    if args.sweep:
        sweep(header, tokens)
    else:
        results = simulate(header, tokens, args.cache_mb, args.policy)
        print_results(results)


if __name__ == '__main__':
    main()
