from core.llm_mind import LLMMind
from config.settings import get_settings

s = get_settings()
client = LLMMind(
    model_name=s.TEST_MODEL,
    api_key=s.TEST_API_KEY,
    base_url=s.TEST_API_URL,
    max_tokens=3200,
    provider="openai",
)

# 直接测试 LLMMind
msgs = [{"role": "user", "content": "你好，请用一句话介绍自己"}]
print("invoke:", client.invoke(msgs))

# 测试流式
print("stream:", end=" ")
for chunk in client.stream_invoke(msgs):
    print(chunk, end="")
print()

# 测试 chat_json
msgs2 = [{"role": "user", "content": "以JSON格式回复: {\"name\": \"AI助手\", \"version\": \"1.0\"}"}]
print("chat_json:", client.chat_json(msgs2))
