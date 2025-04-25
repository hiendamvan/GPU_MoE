Problem: Load imbalance
Test 01: dtype=float32, token=1, model_dim=7168, inter_dim=256, E=256, topk=8
[W425 16:42:12.176075551 collection.cpp:1085] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
[perf] naive moe: 12865.18 us
[perf] custom moe: 744.27 us
[perf] speedup 1628.55%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 02: dtype=float32, token=2, model_dim=7168, inter_dim=256, E=256, topk=8
[perf] naive moe: 14547.45 us
[perf] custom moe: 1490.60 us
[perf] speedup 875.95%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 03: dtype=float32, token=4, model_dim=7168, inter_dim=256, E=256, topk=8
[perf] naive moe: 18486.84 us
[perf] custom moe: 2971.25 us
[perf] speedup 522.19%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 04: dtype=float32, token=8, model_dim=7168, inter_dim=256, E=256, topk=8
[perf] naive moe: 24737.67 us
[perf] custom moe: 5928.90 us
[perf] speedup 317.24%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 05: dtype=float32, token=16, model_dim=7168, inter_dim=256, E=256, topk=8
[perf] naive moe: 36643.67 us
[perf] custom moe: 11789.42 us
[perf] speedup 210.82%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 06: dtype=float32, token=32, model_dim=7168, inter_dim=256, E=256, topk=8
[perf] naive moe: 51507.56 us
[perf] custom moe: 23725.20 us
[perf] speedup 117.10%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 07: dtype=float32, token=32, model_dim=8192, inter_dim=6144, E=8, topk=2
[perf] naive moe: 13287.30 us
[perf] custom moe: 45725.12 us
[perf] speedup -70.94%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 08: dtype=float32, token=128, model_dim=8192, inter_dim=6144, E=8, topk=2
[perf] naive moe: 13793.39 us
[perf] custom moe: 183454.33 us
[perf] speedup -92.48%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 09: dtype=float32, token=32, model_dim=8192, inter_dim=16384, E=8, topk=2
[perf] naive moe: 53568.41 us
[perf] custom moe: 129693.88 us
[perf] speedup -58.70%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
Test 10: dtype=float32, token=128, model_dim=8192, inter_dim=16384, E=8, topk=2
[perf] naive moe: 43176.19 us
[perf] custom moe: 521034.51 us
[perf] speedup -91.71%
[checkAllclose atol=1e-05 rtol=0.001 passed~]
