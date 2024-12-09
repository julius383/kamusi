create-index:
  jq --slurp --compact-output '.' data/swahili-english-dict.jsonl > data/swahili-english-dict.jsonl
  bun run create_index.js data/swahili-english-dict.json

create-db:
  duckdb data/kamusi.db < create_kamusi.sql
