import json
from pathlib import Path
ROOT = Path(r"c:\Users\Roxnnis\Programmation\Python\IA\deep_4_all")
IN = ROOT / "teach_student.json"
OUT1 = ROOT / "train_low_temp_student.json"
OUT2 = ROOT / "train_high_temp_student.json"

data = json.loads(IN.read_text(encoding='utf-8'))
N = len(data)
mid = N // 2
part1 = data[:mid]
part2 = data[mid:]
OUT1.write_text(json.dumps(part1, ensure_ascii=False, indent=4), encoding='utf-8')
OUT2.write_text(json.dumps(part2, ensure_ascii=False, indent=4), encoding='utf-8')
print(f"Total={N}, wrote {len(part1)} to {OUT1} and {len(part2)} to {OUT2}")
