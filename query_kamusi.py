import json
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def run(query):
    sql = f""" 
        SELECT 
            sq.id,
            sq.swahili,
            ka.swahili_alt,
            ka.plural,
            ka.part_of_speech,
            ka.ngeli,
            array_to_json(sq.english) as english,
            array_to_json(ka.examples) as examples,
            array_to_json(ka.alternates) as alternates,
            score
        FROM (
            SELECT *, fts_main_kamusi.match_bm25(
                id,
                '{query}',
                fields := 'english,swahili'
            ) AS score
            FROM kamusi
        ) sq
        INNER JOIN kamusi ka ON ka.id = sq.id
        WHERE score IS NOT NULL
        ORDER BY score DESC;
    """
    p_out = subprocess.run(
        ["duckdb", "-readonly", "-json", "data/kamusi.db", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    if not p_out.stdout:
        print('Nothing found.')
        return
    rs = json.loads(p_out.stdout)
    console = Console()
    for word in rs:
        text = Text()
        for k, v in word.items():
            match (k, v):
                case (_, None):
                    continue
                case (_, []):
                    continue
                case ("swahili", v):
                    text.append(v + " ", style="bold cyan")
                case ("swahili_alt", v):
                    text.append(f"({v}) ", style="bold cyan")
                case ("plural", v):
                    text.append(f"({v}) ", style="bold magenta")
                case ("part_of_speech", v):
                    text.append(
                        f"{v} ",
                        style="italic green",
                    )
                case ("ngeli", v):
                    text.append(f" [{v}] ", style="italic yellow")
                case ("english", v):
                    text.append("\n")
                    for i, w in enumerate(v):
                        text.append(f"    {i + 1}. {w}\n")
                    text.append("\n")
                case ("examples", v):
                    for w in v:
                        text.append(
                            f"        {w[0].replace('~', word['swahili'])} - {w[1]}\n",
                            style="italic",
                        )
        console.print(Panel(text))
    return text


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
