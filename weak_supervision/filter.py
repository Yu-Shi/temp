import json
import copy
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", default="", type=str, help="Input tsv file")
  parser.add_argument("--output_file", default="", type=str, help="Output file")
  args = parser.parse_args()

  output_file = open(args.output_file, 'w', encoding='UTF-8')
  with open(args.input_file, 'r', encoding='UTF-8') as f:
    ln = 0
    wn = 0
    for line in f:
      splitted = line[:-1].split('\t')
      sid = splitted[0]
      queries = splitted[1:]
      last = 0
      modified_queries = []
      for i, query in enumerate(queries):
        last = i
        question = False
        other = False
        if (query.lower().startswith("what") or query.lower().startswith("when") or query.lower().startswith("why") or 
           query.lower().startswith("who") or query.lower().startswith("how") or query.lower().startswith("where") or
           query.lower().startswith("whose") or query.lower().startswith("is") or query.lower().startswith("are") or 
           query.lower().startswith("were") or query.lower().startswith("was") or query.lower().startswith("do") or
           query.lower().startswith("does") or query.lower().startswith("did") or query.lower().startswith("can")):
          question = True
        if query.lower().startswith("tell "):
          other = True
        if (not question) and (not other):
          break
        modified_queries.append(query[0].upper() + query[1:] + ("?" if question == True else "."))
      if last > 1:
        output_file.write(sid + "\t" + "\t".join(modified_queries) + "\n")
        wn += 1
      ln += 1
      
  print(f"total: {ln}, after filtering: {wn}")
  output_file.close()