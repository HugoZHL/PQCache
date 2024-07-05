# 相对于Inf-LLm源码，修改了下面几处：
1. 修改了config： mistral-inf-llm.yaml
2. 修改了pred.py以支持更清楚的配置展示
3. 修改longbench.sh以支持传入压缩参数
4. 修改inf_llm/attention/inf_llm.py, 以支持根据序列长度动态设置topk，local等参数
5. 为了对齐本工作的其他baseline，使用float16（instead of bfloat16推理）
6. 超出max_len时，prompt裁剪方式与LongBench repo中的方案对齐
7. 加入LongBench上对Llama2的评测