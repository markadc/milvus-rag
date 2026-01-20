from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")


def chat(question: str, system_prompt: str, messages: list[dict] = None) -> str:
    """
    调用远程 LLM 获取回答

    Args:
        query: 用户问题
        system: 系统提示词
        messages: 历史消息

    Returns:
        LLM 生成的回答
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=(
            messages
            or [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        ),
        stream=True,
    )
    answer = ""
    for chunk in response:
        c = chunk.choices[0].delta.content
        if c is not None:
            answer += c
            print(c, end="", flush=True)
    print()
    return answer


if __name__ == "__main__":
    q1 = "我叫小白，今年20岁，我是一个程序员，我住在北京。"
    q2 = "小明住在上海"
    system_prompt = """你是一个解析助手，你负责解析人物信息，给出解析结果。
    解析结果的格式为：
    - 姓名：{name}
    - 年龄：{age}
    - 职业：{job}
    - 住址：{address}

    请根据人物信息，给出解析结果。
    """
    chat(q1, system_prompt)
