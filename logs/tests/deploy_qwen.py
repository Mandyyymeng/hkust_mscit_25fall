from vllm import LLM, SamplingParams
import os

# 设置专属环境
os.environ["TMPDIR"] = "/ssddata/zzhanglc/tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# 直接初始化模型引擎
llm = LLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=4096
)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 推理函数
def inference(prompt: str):
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# 测试
if __name__ == "__main__":
    result = inference("Write a Python function to calculate factorial")
    print(result)