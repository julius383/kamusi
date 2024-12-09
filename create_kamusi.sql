-- create sequence serial start 1;
-- nextval('serial') as id,
create or replace table kamusi as 
  select *
  from 
  read_json_auto('data/swahili-english-dict.jsonl', format = 'newline_delimited');

pragma create_fts_index('kamusi', 'id', 'swahili', 'english', overwrite=1);
