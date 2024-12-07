from openai import OpenAI

# 初始化 Moonshot 客户端
client = OpenAI(
    api_key="sk-RxvOfnRQezjNn5tI8Zm9ZSmNPtaD5AmfHS3F14qsiFUdEyDL",
    base_url="https://api.moonshot.cn/v1"
)


def analyze_news_impact(news, industry):
    """
    使用 Moonshot AI 模型分析新闻对行业的影响程度。

    Args:
        news (str): 新闻内容
        industry (str): 行业名称

    Returns:
        str: API 的响应内容，包含新闻影响的小数值
    """
    try:
        # 调用 API
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system",
                 "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，擅长分析新闻内容对行业的影响。"},
                {"role": "user",
                 "content": f"分析以下新闻对行业的影响，并用[-1, 1]的小数表示影响程度：\n\n新闻内容：{news}\n行业：{industry}\n\n请直接输出影响程度的小数值。"}
            ],
            temperature=0.3
        )

        # 提取返回内容
        response_content = completion.choices[0].message.content.strip()
        return response_content

    except Exception as e:
        print(f"调用 API 时出错: {e}")
        return None


# 测试函数
if __name__ == "__main__":
    # key = sk-RxvOfnRQezjNn5tI8Zm9ZSmNPtaD5AmfHS3F14qsiFUdEyDL
    news_example = "某科技公司宣布裁员20%，以降低运营成本。"
    industry_example = "科技行业"

    result = analyze_news_impact(news_example, industry_example)
    print(f"新闻影响分析结果: {result}")
