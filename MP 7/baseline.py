"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def determineMax(tagSet, key, m):
        res = ""
        resCount = 0
        for tag in tagSet:
                if m.get((key, tag), 0) > resCount:
                        res = tag
                        resCount = m.get((key, tag), 0)

        return res

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        m = {}
        tagSet = set()

        tagCount = {}

        seenWords = set()

        result = []

        for sentence in train:
                for (word, tag) in sentence:
                        seenWords.add(word)
                        m[(word, tag)] = m.get((word, tag), 0) + 1
                        tagSet.add(tag)
                        tagCount[tag] = tagCount.get(tag, 0) + 1

        max_key = max(tagCount, key=tagCount.get)

        for sentence in test:
                ans = []
                for word in sentence:
                        if word not in seenWords:
                                ans.append((word, max_key))
                        else:
                                tag = determineMax(tagSet, word, m)
                                ans.append((word, tag))
                result.append(ans)

        return result
                        
