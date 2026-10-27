import json, sys
one, two = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
speedup = one["epoch_time_s"] / two["epoch_time_s"]
print(f"1-node epoch: {one['epoch_time_s']}s | 2-node epoch: {two['epoch_time_s']}s")
print(f"speedup: {speedup:.2f}x | scaling efficiency: {speedup/2*100:.0f}%")
print(f"throughput: {one['avg_tokens_per_s']} -> {two['avg_tokens_per_s']} tok/s")
