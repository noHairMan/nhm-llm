from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-0.8B")
outputs = llm.generate("你好", SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
