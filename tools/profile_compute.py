
from ..vq_method.mistral_profile import FlashMistralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pickle

path = "~/path/to/Mistral-7B-Instruct"
device = 'cuda:0'
config = AutoConfig.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, config=config)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# model = None
model = FlashMistralForCausalLM.from_pretrained(path, config=config)
model = model.half().eval()
model = model.to(device)

with open('prompt.pkl','rb') as fr:
    prompt = pickle.load(fr)

prof_ntime = 10
prev = {}
results = {}
post = {}
for seqlen in range(1000, 21000, 1000):
    input = tokenizer(prompt, truncation=True, max_length=seqlen, return_tensors="pt").to(device)
    model.eval()
    for _ in range(3):
        output = model.generate(
            input_ids=input.input_ids,
            attention_mask=input.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
    model.set_profile(True)
    for _ in range(prof_ntime):
        output = model.generate(
            input_ids=input.input_ids,
            attention_mask=input.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
    model.set_profile(False)
    results[seqlen] = model.all_time / config.num_hidden_layers / prof_ntime
    prev[seqlen] = model.prev_time / prof_ntime
    post[seqlen] = model.post_time / prof_ntime
    print(seqlen, results[seqlen], prev[seqlen], post[seqlen])
import json
with open(f"./result_compute.json","w") as f:
    json.dump(results, f, ensure_ascii = False, indent=4)
with open(f"./result_prevcomp.json","w") as f:
    json.dump(prev, f, ensure_ascii = False, indent=4)
with open(f"./result_postcomp.json","w") as f:
    json.dump(post, f, ensure_ascii = False, indent=4)

