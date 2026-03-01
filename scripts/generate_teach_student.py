import json
from pathlib import Path

ROOT = Path(r"c:\Users\Roxnnis\Programmation\Python\IA\deep_4_all")
RESP = ROOT / "cours" / "TP" / "tp4" / "responses.json"
RESULT = ROOT / "cours" / "TP" / "tp4" / "result.json"
OUT = ROOT / "teach_student.json"

with RESP.open("r", encoding="utf-8") as f:
    responses = json.load(f)
with RESULT.open("r", encoding="utf-8") as f:
    results = json.load(f)

res_map = {r["puzzle_id"]: r for r in results}

teach = []
for resp in responses:
    pid = resp.get("puzzle")
    if pid is None:
        continue
    meta = res_map.get(pid)
    if not meta:
        continue
    solution_san = meta.get("solution_san") or []
    # number_of_my_moves: count of elements at even indices (0,2,4,...)
    number_of_my_moves = (len(solution_san) + 1) // 2
    # enemy_moves: odd-indexed elements (1,3,5,...)
    enemy_moves = " ".join([m for i, m in enumerate(solution_san) if i % 2 == 1])

    initial_fen = meta.get("initial_fen", "")
    instruction = "You are a chess expert and you'll try to answer chess puzzles. You'll be provided with a position in SAN format and should find the best N moves from that position. N is provided by the user."
    input_text = f"I have a chess puzzle for you. The initial position is: {initial_fen}. Try to find a mate or stalemate in {number_of_my_moves} moves in SAN format. The ennemy will do these moves:  {enemy_moves}. Return only the moves in SAN format, without any explanation or commentary. If you can't find a solution, return 'No solution found'."
    output = resp.get("content")

    teach.append({
        "instruction": instruction,
        "input": input_text,
        "output": output
    })

with OUT.open("w", encoding="utf-8") as f:
    json.dump(teach, f, ensure_ascii=False, indent=4)

print(f"Wrote {len(teach)} entries to {OUT}")
