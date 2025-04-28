corpus_path = "/notebooks/embedding/data/raw/ratings.txt"
output_fname = "/notebooks/embedding/data/processed/processed_ratings.txt"
with_label = False

with open(corpus_path, "r", encoding='utf-8') as f1, \
        open(output_fname, 'w', encoding='utf-8') as f2:
    next(f1)
    for line in f1:
        _, sentence, label = line.strip().split('\t')
        if not sentence: continue
        if with_label:
            f2.writelines(sentence + "\u241E" + label + "\n")
        else:
            f2.writelines(sentence + "\n")
            
        