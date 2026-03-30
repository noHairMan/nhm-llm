from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

if __name__ == '__main__':
    # 模型配置
    model_repo = "unsloth/Qwen3.5-0.8B-GGUF"
    model_filename = "Qwen3.5-0.8B-BF16.gguf"  # GGUF 格式的模型文件名
    
    # 缓存目录 - 模型将下载到 ~/.cache/huggingface/hub/
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    
    try:
        print(f"正在从 Hugging Face 下载模型: {model_repo}/{model_filename}")
        
        # 从 Hugging Face 下载模型
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_filename,
            cache_dir=cache_dir,
            resume_download=True  # 支持断点续传
        )
        
        print(f"模型已加载: {model_path}")
        
        # 加载模型
        llm = Llama(model_path=model_path, n_ctx=2048)
        
        # 生成输出
        outputs = llm("你好，请介绍一下 llama", max_tokens=100, temperature=0.8)
        print(outputs)
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示:")
        print(f"- 请确保 {model_repo} 仓库中存在 {model_filename} 文件")
        print("- 访问 https://huggingface.co 查看可用的 GGUF 模型")
        raise
