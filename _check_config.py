"""配置自检 + 引导:核对 .env 里的 key 是否真的能调通百炼。
用法:  python _check_config.py
"""
import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

print("=" * 50)
print("配置自检")
print("=" * 50)

base = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
api_key = os.getenv("OPENAI_API_KEY", "")
dash = os.getenv("DASHSCOPE_API_KEY", "")

print(f"OPENAI_BASE_URL : {base}")
print(f"OPENAI_API_KEY  : {'已设置(长度 ' + str(len(api_key)) + ')' if api_key else '未设置'}")
print(f"DASHSCOPE_API_KEY: {'已设置(长度 ' + str(len(dash)) + ')' if dash else '未设置'}")

# 校验 key 格式
for label, k in (("OPENAI_API_KEY", api_key), ("DASHSCOPE_API_KEY", dash)):
    if not k:
        continue
    ok = bool(re.fullmatch(r"[A-Za-z0-9_\-]+", k)) and not any(ch.isspace() for ch in k)
    print(f"  {label}: {'格式OK' if ok else '格式异常(含空格或非法字符)'}")

# 实测连通性:用 OPENAI_API_KEY(如果有)打百炼的 embedding 端点
if api_key:
    from openai import OpenAI
    try:
        c = OpenAI(api_key=api_key, base_url=base)
        r = c.embeddings.create(model="text-embedding-v4", input=["测试"])
        print(f"  embedding 实测: OK, 向量维度 {len(r.data[0].embedding)}")
    except Exception as e:
        print(f"  embedding 实测: 失败 -> {type(e).__name__}: {str(e)[:120]}")
else:
    print("  OPENAI_API_KEY 未设置,跳过实测")

print("=" * 50)
print("若 OPENAI_API_KEY 未设置或实测失败:")
print("  1. 用你真实的阿里百炼 key(在 https://bailian.console.aliyun.com 获取)")
print("  2. 把它写到 .env 的 OPENAI_API_KEY= 这一行")
print("  3. 重跑本脚本确认")
