from huggingface_hub import list_daily_papers
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"


def extract_pdf(pdf_path, output_path):
    """Extract PDF content to markdown"""
    # TODO: Implement PDF extraction using docling or another high-quality extractor
    pass


def get_daily_papers(output_root, date=None):
    """
    Fetch daily papers from HuggingFace Hub

    Args:
        output_root: Root directory to save papers
        date: Date string in YYYY-MM-DD format (defaults to today)
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    papers = list(list_daily_papers(date=date))
    if len(papers):
        output_dir = output_root / str(date)

        for p in papers:
            params = {"id_list": p.id, "max_results": 1}
            response = requests.get(ARXIV_API_URL, params=params)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entry = root.find("atom:entry", ns)

                if entry is not None:
                    paper_dir = output_dir / p.id
                    paper_dir.mkdir(parents=True, exist_ok=True)

                    pdf_response = requests.get(ARXIV_PDF_URL.format(id=p.id))
                    if pdf_response.status_code == 200:
                        pdf_file = paper_dir / "paper.pdf"
                        pdf_file.write_bytes(pdf_response.content)
                        print(f"Downloaded: {p.id} - {p.title}")

                        md_file = paper_dir / "paper.md"
                        extract_pdf(pdf_file, md_file)
                        print(f"Extracted: {p.id}")
                    else:
                        print(
                            f"Failed to download PDF for {p.id}: {pdf_response.status_code}"
                        )

                    metadata = {
                        "id": p.id,
                        "title": p.title,
                        "paper": p.paper if hasattr(p, 'paper') else None,
                    }
                    for attr in dir(p):
                        if not attr.startswith('_') and attr not in ['id', 'title', 'paper']:
                            value = getattr(p, attr, None)
                            if not callable(value):
                                if isinstance(value, datetime):
                                    metadata[attr] = value.isoformat()
                                else:
                                    metadata[attr] = value

                    metadata_file = paper_dir / "metadata.json"
                    metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
                else:
                    print(f"No entry found for {p.id}")
            else:
                print(f"API request failed for {p.id}: {response.status_code}")
