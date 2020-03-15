import json
import copy
import spacy
import tqdm
import random
import argparse

nlp = spacy.load("en_core_web_sm")

question_words = ["what", "when", "why", "who", "how", "where", "whose", "is", "are", "were", "was", "do", "does", "did"]

def include(npset: set, np: str):
  if np in npset:
    return True
  for n in npset:
    if n.find(np) != -1:
      return True
  return False


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", default="", type=str, help="Input tsv file (filtered)")
  parser.add_argument("--output_file", default="", type=str, help="Output file (NDJson, for GPT-2)")
  parser.add_argument("--do_replacement", action='store_true', help="Whether to replace noun chunks with pronouns")
  parser.add_argument("--do_omission", action='store_true', help="Whether to omit the structure of preposition + noun")
  args = parser.parse_args()

  if not (args.do_omission or args.do_replacement or args.zero):
    print("error")
    exit(0)

  output_file = open(args.output_file, 'w', encoding='UTF-8')
  random.seed(42)
  with open(args.input_file, 'r', encoding='UTF-8') as f:
    ln = 0
    all_lines = f.readlines()
    for line in tqdm.tqdm(all_lines):
      splitted = line[:-1].split('\t')
      sid = splitted[0]
      queries = splitted[1:]
      last_nchks_set = set()
      this_nchks_set = set()
      last_tokens = []
      this_tokens = []
      modified_queries = []
      for i, query in enumerate(queries):
        doc = nlp(query)

        char_to_word = []
        word_to_char = []
        pos = 0
        for token in doc:
          this_tokens.append(token.text)
          pos = query.find(token.text, pos)
          assert pos != -1
          word_to_char.append((pos, pos + len(token)))
          
          for c in range(len(token)):
            char_to_word.append(token.i)
          while len(query) > len(char_to_word) and query[len(char_to_word)] in [' ']:
            char_to_word.append(token.i)

        if len(char_to_word) != len(query):
          break  # something weird happens!

        new_query = copy.deepcopy(query)

        noun_chunks = list(doc.noun_chunks)
        no_new = True
        replace = False

        if not replace:
          for chk in noun_chunks:
            contain_stop = False
            for stop in question_words:
              if chk.text.lower().find(stop) != -1:
                contain_stop = True
                break
            if contain_stop:
              continue

            chk_cleaned = chk
            article = None
            if chk.end - chk.start > 1 and chk[0].text in ['a', 'an', 'the']:
              chk_cleaned = chk[1:]   # 去掉冠词后
              article = chk[0].text   # 冠词
            print(chk_cleaned)
            
            # 去掉冠词进行考察
            if include(last_nchks_set, chk_cleaned.text.lower()):

              print(last_nchks_set)
              last_word_pos = chk.end - 1

              if replace == False and doc[last_word_pos].tag_ in ["NN", "NNP"]:  # singular

                word_pos = chk.start
                pre = word_pos - 1
                rw = "it"               # it+its: 123, he+his+him: 3, she+her+hers: 4
                r = random.random()
                if r <= 0.02:
                  rw = "he"
                elif r >= 0.98:
                  rw = "she"

                if args.do_omission and pre >= 0 and doc[pre].tag_ in ["IN"]:
                  pre_span = word_to_char[pre]
                  start = pre_span[0] - 1 if pre_span[0] > 0 else pre_span[0]
                  new_query = new_query[:start] + new_query[pre_span[1]+1:]
                  new_query = new_query.replace(chk.text, "")
                  replace = True
                elif args.do_replacement:
                  new_query = new_query.replace(chk.text, rw)
                  replace = True

              elif replace == False and doc[last_word_pos].tag_ in ['NNS', 'NNPS']:  # plural

                word_pos = chk.start
                pre = word_pos - 1

                if args.do_omission and pre >= 0 and doc[pre].tag_ in ["IN"]:
                  pre_span = word_to_char[pre]
                  start = pre_span[0] - 1 if pre_span[0] > 0 else pre_span[0]
                  new_query = new_query[:start] + new_query[pre_span[1]+1:]
                  new_query = new_query.replace(chk.text, "")
                  replace = True
                elif args.do_replacement:
                  r = random.randint(1, 4)
                  if r == 4:  # 1/4 chance. they: 34, them: 12
                    new_query = new_query.replace(chk.text, "them")
                  else:       # 3/4 chance.
                    new_query = new_query.replace(chk.text, "they")
                  replace = True

            else:
              this_nchks_set.add(chk_cleaned.text.lower())
              no_new = False

        if no_new:
          this_nchks_set = copy.deepcopy(last_nchks_set)
        last_nchks_set = copy.deepcopy(this_nchks_set)
        this_nchks_set.clear()
        modified_queries.append(new_query)

        if replace:
          line = {"topic_number": sid, "query_number": i + 1, "input": modified_queries, "target": query}
          output_file.write(json.dumps(line) + '\n')

      ln += 1

  output_file.close()
