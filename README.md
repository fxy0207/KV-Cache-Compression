# KV-Cache-Compression
KV Cache Compression - FYP Project README  
Project: Efficient Large Language Models: KV Cache Compression  
Student: FENG Xiao Yu  


## Project Overview
This project implements and evaluates KVzip, a query-agnostic KV cache compression algorithm for Transformer-based large language models (LLMs). KVzip achieves:

3–4× KV cache size reduction

~2× FlashAttention decoding latency reduction

Near-lossless accuracy across diverse tasks (QA, retrieval, reasoning, code comprehension)

The method is query-agnostic: compression happens once per context and the compressed cache can be reused for multiple diverse queries without recomputation.

Key Insight: A compressed KV cache that enables the model to reconstruct the original context will also serve diverse downstream queries effectively.


## Current Progress
Area	Status	Details
Literature Review	✅ Complete	Reviewed H₂O, SnapKV, PyramidKV, StreamingLLM, DuoAttention  
Project Statement	✅ Complete	Formal statement approved  
Environment Setup	✅ Complete	Python 3.10, CUDA 12.1 configured  
KVzip Repository	✅ Cloned	[https://github.com/snu-mllab/KVzip]  
Baseline Understanding	✅ Complete	Familiar with implementation principles  


## Quick Start
```python
from model import ModelKVzip

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
context = "This is my basic profile. My name is Kim living in Seoul. My major is computer science."
queries = ["What is my name?", "Do I live in Seoul?"]

kv = model.prefill(context, load_score=False)  # prefill KV cache + importance scoring
kv.prune(ratio=0.3)  # compression ratio, evict 70% KV

for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(q, output)
```


## Two-Week Plan (Weeks 1–2)
Task	Description	Priority	Est. Time
1. Environment Setup	Install dependencies, FlashAttention 2.7.4	High	2 hrs
2. Model Loading Test	Verify Qwen2.5-7B & LLaMA3.1-8B loading	High	3 hrs
3. Run Demo	Execute demo.py end-to-end	High	2 hrs
4. Baseline Implementation	Set up H₂O, SnapKV, PyramidKV in query-agnostic mode	High	6 hrs
5. Evaluation Harness	Build modular pipeline for SQuAD, GSM8K, NIAH	Medium	4 hrs
6. Documentation	Update README with findings	Medium	2 hrs
