import math
from collections import defaultdict
import re

class PositionalIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list)) 
        self.documents = {} 
        self.doc_lengths = {}  
    
    def clean_text(self, text):
        """Removes punctuation and possessive 's from the text."""
        text = text.lower()
        text = re.sub(r"\b\w+'s\b", '', text)
        text = re.sub(r"[.,!\-]", '', text)  # Remove ., !, - characters
        return text

    def add_document(self, doc_id, text):
        text = self.clean_text(text)  # Clean text before processing
        words = text.split()
        self.documents[doc_id] = text  # Store cleaned text
        self.doc_lengths[doc_id] = len(words)
        for pos, word in enumerate(words):
            self.index[word][doc_id].append(pos)

    def word_search(self, word):
        return {doc_id: positions for doc_id, positions in self.index.get(word, {}).items()}

    def word_search_in_doc(self, word, doc_id):
        return self.index.get(word, {}).get(doc_id, [])

    def proximity_search(self, word1, word2, max_distance):
        word1 = word1.lower()
        word2 = word2.lower()
        
        results = set()

        if word1 not in self.index or word2 not in self.index:
            return results  
        
        for doc_id in self.index[word1]:
            if doc_id in self.index[word2]: 
                positions1 = sorted(self.index[word1][doc_id])  
                positions2 = sorted(self.index[word2][doc_id])

                i, j = 0, 0
                while i < len(positions1) and j < len(positions2):
                    if abs(positions1[i] - positions2[j]) <= max_distance:
                        results.add(doc_id)
                        break  
                    if positions1[i] < positions2[j]:
                        i += 1
                    else:
                        j += 1

        return results

    def phrase_search(self, phrase):
        words = phrase.split()
        if not words:
            return set()
        possible_docs = set(self.index.get(words[0], {}).keys())
        for word in words[1:]:
            possible_docs &= set(self.index.get(word, {}).keys())
        results = []
        for doc_id in possible_docs:
            positions = [self.index[word][doc_id] for word in words]
            for p in zip(*positions):
                if all(p[i] + 1 == p[i + 1] for i in range(len(p) - 1)):
                    results.append(doc_id)
                    break
        return set(results)

    def tf_idf(self, word, doc_id):
        tf = len(self.index.get(word, {}).get(doc_id, [])) / self.doc_lengths[doc_id]
        df = len(self.index.get(word, {}))
        idf = math.log((1 + len(self.documents)) / (1 + df)) + 1
        return tf * idf

    def bm25(self, word, doc_id, k1=1.5, b=0.75):
        tf = len(self.index.get(word, {}).get(doc_id, []))
        doc_len = self.doc_lengths[doc_id]
        avg_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        df = len(self.index.get(word, {}))
        idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1)
        return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))

pos_index = PositionalIndex()
pos_index.add_document(1, "Ducks are a group of species of water birds, relatively small in size and with shorter necks than their close cousins of swans and geese. Along with their longer necked cousins, they make up the biological family of Anatidae. Humans have had a long relationship with members of this family by being economically and culturally important to us. Ducks, in particular, have been domesticated to be exploited for their feathers, eggs and meat.")
pos_index.add_document(2, "Over the hills, behind the windmills by the river lived a pregnant duck. A beautiful duck inside and out. In a few days she will be laying her eggs. The duck hoped for beautiful and smart ducklings just like herself, but since her husband was an average duck she was not sure how her ducklings would turn out. After days of waiting the beautiful duck’s eggs hatched. Six tiny ducklings were born, not as beautiful as their mother and not as average as their father. The six ducklings were not good looking at all. The mother duck knew her ducklings will not get far with their looks and hoped they got her intelligent mind.")

# print("Word Search (duck):", pos_index.word_search("duck"))
# print("Proximity Search (duck, bird, max_distance=10):", pos_index.proximity_search("ducks", "birds", 100))
# print("Phrase Search ('water birds'):", pos_index.phrase_search("water birds"))
# print("TF-IDF (duck, doc 1):", pos_index.tf_idf("duck", 1))
# print("BM25 (duck, doc 1):", pos_index.bm25("duck", 1))
print("Word 1: ")
word_input_1 = input()
print("Word 2: ")
word_input_2 = input()
print(f"Proximity Search between \"{word_input_1}\" and \"{word_input_2}\" is: ", pos_index.proximity_search(word_input_1, word_input_2, 10))
print(pos_index.word_search(word_input_1))
print(pos_index.word_search(word_input_2))