from huggingface_hub import list_daily_papers
from datetime import datetime


def get_daily_papers():
    """Fetch daily papers from HuggingFace Hub"""
    today = datetime.today().strftime("%Y-%m-%d")
    return list_daily_papers(date=today)


if __name__ == "__main__":
    papers = get_daily_papers()
    for p in papers:
        print(p)
