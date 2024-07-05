## We made only four modifications:
1. Transfer KV Cache backup storage to CPU RAM 
2. Removed the version restriction for transformers
3. When calculating the average of value vectors, we switched to using float32
4. Added the sink token trick