import os
import sys
import dotenv

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print(os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
print(os.getenv("OPENAI_API_KEY", "ollama"))

# 1、获取对话模型：
chat_model = ChatOpenAI(
    #必须要设置的3个参数
    model_name="qwen2.5-coder:1.5b",   #默认使用的是qwen2.5-coder:1.5b模型
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],

)

# 2、调用模型
response = chat_model.invoke("什么是langchain?")

# 3、查看响应的文本
print(response.content)
