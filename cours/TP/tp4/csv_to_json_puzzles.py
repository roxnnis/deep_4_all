#!/usr/bin/env python3
"""Extraire 1000 puzzles aléatoires depuis un CSV et écrire result.json

Le CSV attendu a l'en-tête :
PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Comportement :
- Lire toutes les lignes, mélanger et essayer d'extraire jusqu'à `--total` puzzles valides
- Pour chaque ligne :
  - `FEN` est la position AVANT le coup de l'adversaire
  - appliquer le 1er coup (UCI) à la FEN pour obtenir la position initiale à présenter
  - la solution commence au 2e coup (les coups restants)
  - convertir les coups UCI en SAN pour affichage
- Sortie JSON : liste d'objets {"puzzle_id","initial_fen","solution_uci","solution_san"}

Usage:
  python csv_to_json_puzzles.py --csv puzzles.csv --out result.json --total 1000

Dépendance: `python-chess` (pip install python-chess)
"""

import csv
import json
import random
import argparse
import sys

try:
    import chess
except Exception:
    print("Le module 'python-chess' est requis. Installez-le avec: pip install python-chess")
    sys.exit(1)


def parse_moves_field(moves_field: str):
    if not moves_field:
        return []
    # séparer par espaces ou virgules
    parts = []
    for sep in ["\n", ",", ";"]:
        moves_field = moves_field.replace(sep, " ")
    for tok in moves_field.split():
        tok = tok.strip()
        if tok:
            parts.append(tok)
    return parts


def process_row(row):
    pid = row.get("PuzzleId") or row.get("puzzleid") or row.get("id")
    fen = row.get("FEN") or row.get("fen")
    moves_field = row.get("Moves") or row.get("moves") or ""
    uci_moves = parse_moves_field(moves_field)
    if not pid or not fen or len(uci_moves) < 2:
        return None

    board = chess.Board(fen)
    try:
        first = chess.Move.from_uci(uci_moves[0])
    except Exception:
        return None
    if first not in board.legal_moves:
        # si le premier coup n'est pas légal sur la FEN, abandon
        return None
    board.push(first)
    initial_fen = board.fen()

    solution_uci = []
    solution_san = []
    # la solution commence au second coup
    for uci in uci_moves[1:]:
        try:
            m = chess.Move.from_uci(uci)
        except Exception:
            return None
        if m not in board.legal_moves:
            return None
        san = board.san(m)
        solution_uci.append(uci)
        solution_san.append(san)
        board.push(m)

    if not solution_uci:
        return None

    return {
        "puzzle_id": pid,
        "initial_fen": initial_fen,
        "solution_uci": solution_uci,
        "solution_san": solution_san,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data.csv", help="Chemin vers le CSV source")
    parser.add_argument("--out", default="result.json", help="Fichier JSON de sortie")
    parser.add_argument("--total", type=int, default=1000, help="Nombre de puzzles à extraire")
    parser.add_argument("--seed", type=int, default=None, help="Graine pour reproductibilité")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    rows = []
    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("Aucune ligne trouvée dans le CSV.")
        return

    random.shuffle(rows)

    results = []
    seen_ids = set()
    for row in rows:
        if len(results) >= args.total:
            break
        out = process_row(row)
        if out is None:
            continue
        if out["puzzle_id"] in seen_ids:
            continue
        results.append(out)
        seen_ids.add(out["puzzle_id"])

    print(f"Sélectionnés {len(results)} puzzles (demandé {args.total})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Écrit ->", args.out)


if __name__ == "__main__":
    main()
