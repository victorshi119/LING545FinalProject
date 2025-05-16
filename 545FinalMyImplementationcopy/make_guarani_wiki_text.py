# make_guarani_wiki_txt.py
# make_guarani_wiki_text.py
# make_guarani_wiki_text.py
import os

input_dir = "Guarani_extracted"
output_file = "guarani_wiki.txt"

with open(output_file, "w", encoding="utf-8") as out_f:
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.startswith("wiki_"):  # Match wiki_00, wiki_01, etc.
                full_path = os.path.join(root, fname)
                with open(full_path, encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line.replace("\n", " ") + "\n")



# import os
# import json

# input_dir = "Guarani_extracted"
# output_file = "guarani_wiki.txt"

# with open(output_file, "w", encoding="utf-8") as out_f:
#     for root, dirs, files in os.walk(input_dir):
#         for fname in files:
#             if fname.endswith(".json"):
#                 with open(os.path.join(root, fname), encoding="utf-8") as in_f:
#                     for line in in_f:
#                         try:
#                             obj = json.loads(line)
#                             text = obj.get("text", "").strip()
#                             if text:
#                                 out_f.write(text.replace("\n", " ") + "\n")
#                         except json.JSONDecodeError:
#                             continue
