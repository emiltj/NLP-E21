
# Import libraries and packages
import os, re
from pathlib import Path

# Pt. 1

# 1.1 Loading in the texts
# Import corpus
corpus = []

for filename in Path("/work/Coder Python/nlp_assignment_1-30348bb8/nlp_exercises/NLP-E21/syllabus/classes/class2/train_corpus").glob("*.txt"): # is there a way to take a path in a more elegant way?
    with open(filename, "r", encoding = "utf-8") as file:
        loaded_text = file.read()
        corpus.append(loaded_text)

# 1.2 Segment sentences in the texts
corpus_split = [i.split(". ") for i in corpus]

corpus_split[0]

# 1.3 Tokenize each sentence
for textt in corpus_split:
    for sentencee in textt:
        re.split('[^a-zA-Z]', sentencee)

# For text in corpus_split, for sentence in text, return sentence, split
corpus_token = [re.split('[^a-zA-Z]', sentence) for text in corpus_split for sentence in text]


# Part 2

# 2.1

