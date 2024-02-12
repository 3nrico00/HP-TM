import re

def read_hp(title_path):
    with open(title_path, "r", encoding="utf8") as book:
        lines = [line.strip() for line in book.readlines() if not (line.startswith("Page |") or line.strip() == '')]
    # Join the lines into a single string
    text = '\n'.join(lines)
    text = text.replace("\n", " ").replace("\r", "").replace("CHPT", "")
    
    return text

class CustomTokenizer:
    def __init__(self):
        self.patterns = [
            #(r'\w+', 'WORD'),         # Matches words
            #(r'\d+', 'NUMBER'),        # Matches numbers
            #(r'[.,;!?]', 'PUNCTUATION'),  # Matches common punctuation
            #(r"\b(?:\w+'\w*|\w+?n't)\b", 'abbreviation')
            (r"\b\w+'t\b|\b\w+\b|'\w+\b", "WORD")
            # (r'\b[A-Za-z]+\.(?![a-z])', 'WORD')  # Matches sequences of capitalized words (potential sentences)
            # (r'\b(?:[A-Za-z]+\.?\'?[A-Za-z]*|\w+)\b', 'WORD'),
            #(r"\b(?:\w+'\w*|(?<!\w)'(?:t|re|s|m|ll|ve)\b|\w+)\b", 'WORD')
        ]

    def tokenize(self, text):
        tokens = []
        for pattern, token_type in self.patterns:
            regex = re.compile(pattern)
            matches = regex.finditer(text)
            for match in matches:
                tokens.append((match.group(), token_type))
        return tokens