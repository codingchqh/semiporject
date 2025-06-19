import openai
import os
from dotenv import load_dotenv
from openai import OpenAI



def summarize_text(text: str) -> str:
    prompt = f"다음 내용을 요약해 주세요:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 또는 "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

class GPTSummarizer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful summarizer."},
                {"role": "user", "content": f"다음 내용을 요약해줘:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content

# 사용 예시
if __name__ == "__main__":
    sample_text = "여기에 요약할 긴 텍스트를 넣으세요."
    # 함수 사용
    summary1 = summarize_text(sample_text)
    print("함수 요약 결과:\n", summary1)

    # 클래스 사용
    api_key = os.getenv("OPENAI_API_KEY")
    summarizer = GPTSummarizer(api_key)
    summary2 = summarizer.summarize(sample_text)
    print("클래스 요약 결과:\n", summary2)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
summarizer = GPTSummarizer(api_key="sk-로시작하는 api키를 입력")