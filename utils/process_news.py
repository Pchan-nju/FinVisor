import json
from utils.news_impact import analyze_news_impact  # 导入您在 news_impact.py 中定义的函数


# 读取新闻数据
def load_news_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# 保存结果到 JSON 文件
def save_news_to_json(news_list, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(news_list, file, ensure_ascii=False, indent=4)


# 处理新闻数据，调用影响分析函数并保存结果
def process_news(input_file, output_file):
    # 加载原始新闻数据
    news_data = load_news_from_json(input_file)
    processed_data = []

    for news in news_data:
        # 提取必要字段
        content = news.get("content", "")
        industry = news.get("industry", "")
        date = news.get("date", "")

        # 调用影响分析函数生成影响评分
        try:
            impact_score = float(analyze_news_impact(content, industry))  # 确保返回值为浮点数
        except Exception as e:
            print(f"Error processing news: {news}. Error: {e}")
            impact_score = None  # 如果发生错误，影响评分设为 None

        # 构建带有影响评分的结果
        processed_data.append({
            "content": content,
            "industry": industry,
            "date": date,
            "impact_score": impact_score
        })

    # 保存处理后的结果
    save_news_to_json(processed_data, output_file)
    print(f"Processed data saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 文件路径
    input_file = "../data/news_data.json"  # 原始新闻数据文件
    output_file = "../data/news_impact_data.json"  # 带影响评分的新闻数据文件

    # 执行新闻处理
    process_news(input_file, output_file)
