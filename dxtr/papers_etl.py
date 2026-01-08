"""
Papers ETL Pipeline

End-to-end pipeline for retrieving and processing research papers:
1. Download papers from HuggingFace
2. Convert PDFs to markdown (using local Docling)
3. Generate embeddings and build index
"""

import json
from pathlib import Path
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from dxtr.config_v2 import config
from dxtr.util import get_daily_papers
from dxtr.docling_utils import convert_pdf_local, generate_embeddings_local


class PapersETL:
    """End-to-end pipeline for paper retrieval and processing"""

    def __init__(self, output_dir: Path = None):
        """
        Initialize the ETL pipeline

        Args:
            output_dir: Where to save papers (defaults to config.paths.papers_dir)
        """
        self.output_dir = output_dir or config.paths.papers_dir

    def run(self, date: str = None, max_papers: int = None):
        """
        Run the complete ETL pipeline

        Args:
            date: Date string in YYYY-MM-DD format (defaults to today)
            max_papers: Maximum papers to process (None = all)
        """
        # Default to today
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        print(f"Papers ETL Pipeline - {date}")

        # Step 1: Download papers
        print(f"Downloading papers from HuggingFace...")
        papers = self._download_papers(date, max_papers)

        if not papers:
            print("  No papers found")
            return

        print(f"  ✓ Downloaded {len(papers)} paper(s)\n")

        # Step 2: Convert PDFs to markdown + build indices
        print(f"Processing papers...")
        self._convert_papers(papers)

        print()
        print("Pipeline complete")

    def _download_papers(self, date: str, max_papers: int = None) -> list[Path]:
        """Download papers and return list of PDF paths"""
        papers = get_daily_papers(output_root=self.output_dir, date=date)

        # Collect all paper.pdf files
        date_dir = self.output_dir / date
        if not date_dir.exists():
            return []

        pdf_files = list(date_dir.glob("*/paper.pdf"))

        if max_papers:
            pdf_files = pdf_files[:max_papers]

        return pdf_files

    def _convert_papers(self, pdf_files: list[Path]):
        """Convert all PDFs to markdown and LlamaIndex format"""
        successful = 0
        failed = 0
        skipped = 0

        for idx, pdf_path in enumerate(pdf_files, 1):
            paper_id = pdf_path.parent.name
            md_path = pdf_path.parent / "paper.md"
            layout_path = pdf_path.parent / "layout.json"
            index_dir = pdf_path.parent / "paper.index"

            # Skip if already processed
            if md_path.exists() and layout_path.exists() and index_dir.exists():
                print(f"  [{idx}/{len(pdf_files)}] {paper_id}: Already processed")
                skipped += 1
                continue

            print(f"  [{idx}/{len(pdf_files)}] {paper_id}:")

            try:
                # Step 1: Convert PDF to markdown + layout JSON
                print(f"    Converting to markdown...", end=" ", flush=True)
                
                # Use local Docling conversion
                result = convert_pdf_local(pdf_path)

                md_path.write_text(result["markdown"], encoding="utf-8")
                
                layout_path.write_text(
                    json.dumps(result["docling_json"], indent=2), encoding="utf-8"
                )
                print("✓")

                # Step 2: Build index from markdown
                print(f"    Building index:")
                document = Document(text=result["markdown"])

                print(f"      Parsing nodes...", end=" ", flush=True)
                
                node_parser = SentenceSplitter(
                    chunk_size=1024,  # ~750 tokens
                    chunk_overlap=128,
                )
                nodes = node_parser.get_nodes_from_documents([document])
                print(f"{len(nodes)} nodes")

                # Generate embeddings using local model
                print(f"      Generating embeddings...", end=" ", flush=True)
                texts = [node.get_content() for node in nodes]
                embeddings = generate_embeddings_local(texts)

                # Assign embeddings to nodes
                for node, embedding in zip(nodes, embeddings):
                    node.embedding = embedding
                print("✓")

                # Build index with pre-computed embeddings (no embed_model needed)
                print(f"      Creating index...", end=" ", flush=True)
                index = VectorStoreIndex(nodes, embed_model=None)
                print("✓")

                print(f"      Persisting...", end=" ", flush=True)
                index.storage_context.persist(persist_dir=str(index_dir))
                print("✓")

                successful += 1

            except Exception as e:
                print(f"✗ {e}")
                failed += 1

        # Summary
        print()
        print(f"  Results: {successful} successful, {skipped} skipped, {failed} failed")


# Convenience function for CLI
def run_etl(date: str = None, max_papers: int = None):
    """
    Run the papers ETL pipeline

    Args:
        date: Date string in YYYY-MM-DD format (defaults to today)
        max_papers: Maximum papers to process (None = all)
    """
    etl = PapersETL()
    etl.run(date=date, max_papers=max_papers)