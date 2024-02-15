import re
import numpy as np
from collections import Counter
import re
from scipy import stats
from numpy.linalg import norm
import time


def read_hp(title_path):
    with open(title_path, "r", encoding="utf8") as book:
        lines = [line.strip() for line in book.readlines() if not line.startswith("Page |")]
    # Join the lines into a single string
    text = '\n'.join(lines)
    text = text.replace("\n", " ").replace("\r", "").replace("CHPT", "")
    
    return text

class CustomTokenizer:
    def __init__(self):
        self.patterns = [
            (r'[.,;!?]', 'PUNCTUATION'),  # Matches common punctuation
            (r"\b\w+'t\b|\b\w+\b|'\w+\b", "WORD") #specific pattern for words
        ]

    def tokenize(self, text):
        tokens = []
        for pattern, token_type in self.patterns:
            regex = re.compile(pattern)
            matches = regex.finditer(text)
            for match in matches:
                tokens.append((match.group(), token_type))
        return tokens

def preproc(text, custom_tokenizer, stop_words): 
    text = text.lower()
    text = custom_tokenizer.tokenize(text)
    text = [i[0] for i in text if i[1] == 'WORD']
    text = [i for i in text if i not in stop_words]
    text = [word for word in text if not word.startswith("'")]
    text = [i for i in text if len(i)>1]
    return text

class counter:
    def __init__(self, lst):
        self.counts = {}
        self._count(lst)

    def _count(self, lst):
        for el in lst:
            if el in self.counts:
                self.counts[el] += 1
            else:
                self.counts[el] = 1

    def get_counts(self):
        return self.counts

    def most_common(self, n=None):
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

        if n is not None:
            return sorted_counts[:n]
        else:
            return dict(sorted_counts)

