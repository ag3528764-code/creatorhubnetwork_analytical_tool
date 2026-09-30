import json
from pathlib import Path
def save_json(data,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
