import json
import openai
from trafilatura import fetch_url, extract
#API 配置
openai.api_key=''
openai.api_base = "https://lonlie.plus7.plus/v1"
gpt_model = "gpt-4-0613"
def text_from_web(url, output_format='txt'):
    # url = 'https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/'
    downloaded = fetch_url(url)
    result = extract(downloaded, output_format=output_format)
    # print(result)
    return result

def ask_question(html_content):
    response = openai.ChatCompletion.create(
        model = gpt_model,
        messages=[
            {"role": "system", "content": "You are a chatbot."},
            {"role": "user", "content": f"从提供的html页面中提取完整地新闻内容，忽略所有广告及导航页。HTML内容：{html_content}"}
        ],
        temperature=0.7
    )
    #answer = response.choices[0].message.content
    answer = response['choices'][0]['message']['content']
    return answer

def process_html_file(soup):
    extracted_content = ask_question(soup)
    return {
        'file': url,
        'extracted_content': extracted_content
    }

def save_results_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    url = 'https://news.sina.com.cn/s/2023-11-20/doc-imzvfrzw9573562.shtml'#要提取内容的网站url
    output_file = '/home/zyy/tb/news/output.json'  # 输出路径
    news = text_from_web(url)
    print(news)
    results = []
    results.append(process_html_file(news)) 
    save_results_to_json(results, output_file)