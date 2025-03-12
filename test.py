import spacy
import math
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")
class PositionalIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))  
        self.documents = {}
        self.doc_lengths = {}

    def preprocess_text(self, text):
        doc = nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha]
        return words

    def add_document(self, doc_id, text):
        words = self.preprocess_text(text)
        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(words)

        for pos, word in enumerate(words):
            self.index[word][doc_id].append(pos)

    def word_search(self, word):
        word = word.lower()
        return {doc_id: positions for doc_id, positions in self.index.get(word, {}).items()}

    # def word_search_in_doc(self, word, doc_id):
    #     word = word.lower()
    #     return self.index.get(word, {}).get(doc_id, [])

    def proximity_search(self, word1, word2, max_distance):
        word1, word2 = word1.lower(), word2.lower()
        results = {}

        if word1 not in self.index or word2 not in self.index:
            return results 

        for doc_id in self.index[word1]:
            if doc_id in self.index[word2]: 
                positions1 = sorted(self.index[word1][doc_id]) 
                positions2 = sorted(self.index[word2][doc_id]) 
                i, j = 0, 0
                position_pairs = [] 
                while i < len(positions1) and j < len(positions2):
                    if abs(positions1[i] - positions2[j]) <= max_distance:
                        position_pairs.append((positions1[i], positions2[j]))  
                        i += 1 
                        j += 1  
                    elif positions1[i] < positions2[j]:
                        i += 1
                    else:
                        j += 1
                if position_pairs:
                    results[doc_id] = position_pairs  

        return results  

    def phrase_search(self, phrase):
        words = self.preprocess_text(phrase)
        if not words:
            return {}

        possible_docs = set(self.index.get(words[0], {}).keys())  
        for word in words[1:]:
            possible_docs &= set(self.index.get(word, {}).keys())  

        results = {}

        for doc_id in possible_docs:
            word_positions = [self.index[word][doc_id] for word in words]  
            phrase_positions = []

            for positions in zip(*word_positions):
                if all(positions[i] + 1 == positions[i + 1] for i in range(len(positions) - 1)):
                    phrase_positions.append(positions[0])  

            if phrase_positions:
                results[doc_id] = phrase_positions  

        return results 


    def tf_idf(self, word, doc_id):
        word = word.lower()
        tf = len(self.index.get(word, {}).get(doc_id, [])) / self.doc_lengths[doc_id]
        df = len(self.index.get(word, {}))
        idf = math.log((1 + len(self.documents)) / (1 + df)) + 1
        return tf * idf

    def bm25(self, word, doc_id, k1=1.5, b=0.75):
        word = word.lower()
        tf = len(self.index.get(word, {}).get(doc_id, []))
        doc_len = self.doc_lengths[doc_id]
        avg_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        df = len(self.index.get(word, {}))
        idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1)
        return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))
pos_index = PositionalIndex()

pos_index.add_document(1, "Regular exercise offers numerous benefits, both physically and mentally. Engaging in physical activities like jogging, swimming, or cycling helps improve cardiovascular health, strengthen muscles, and enhance flexibility. Exercise also plays a crucial role in weight management and reducing the risk of chronic diseases such as diabetes and hypertension. Moreover, regular physical activity is known to boost mental health by alleviating symptoms of depression and anxiety, promoting better sleep, and enhancing overall mood. By making exercise a part of your daily routine, you can significantly improve your quality of life and longevity.")
pos_index.add_document(2, "Effective time management is essential for achieving success and maintaining a balanced life. By prioritizing tasks and setting clear goals, individuals can maximize productivity and reduce stress. Time management involves creating schedules, setting deadlines, and avoiding procrastination. It also requires the ability to delegate tasks when necessary and to focus on one task at a time. Proper time management not only helps in completing tasks efficiently but also allows for more free time to relax and pursue personal interests. Developing good time management skills is key to achieving both professional and personal goals.")
pos_index.add_document(3, "Social media has revolutionized the way we communicate and interact with one another. Platforms like Facebook, Twitter, and Instagram have made it easier to stay connected with friends and family, share information, and express opinions. However, social media also has its downsides. It can lead to issues such as cyberbullying, privacy concerns, and the spread of misinformation. Additionally, excessive use of social media can contribute to mental health problems like anxiety and depression. It is important for users to be mindful of their social media habits and strive for a healthy balance between online and offline interactions.")

# word_search = input("Enter your word search: ")
# print(f"Word Search {word_search}:", pos_index.word_search(word_search))

# distance = 10
# dis_search = input(f"Enter two words you want to search with distance={distance}: ")
# [word1, word2] = dis_search.split()
# print(f"Proximity Search \"{word1}\", \"{word2}\", distance: {distance}: ", pos_index.proximity_search(word1, word2, distance))

# pharse = input("Enter your pharse word: ")
# print(f"Phrase Search \"{pharse}:\"", pos_index.phrase_search(pharse))

word = input("Enter your word ranking document: ")
document = int(input("Enter your document id: "))
print(f"TF-IDF \"{word}\", \"{document}\": ", pos_index.tf_idf(word, document))
print(f"BM25 \"{word}\", \"{document}\": ", pos_index.bm25(word, document))

