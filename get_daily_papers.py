from pathlib import Path
from dxtr import util

output_root = Path(".dxtr/hf_papers")

util.get_daily_papers(output_root)
