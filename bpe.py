from collections import defaultdict, Counter

class BPE:
    def __init__(self, corpus: list[str], vocab_size: int,
                 max_iter: int | None = None,
                 debug: bool = False
                 ):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = []
        self.max_iter = max_iter
        self.word_freq = Counter()
        self.splits = {}                          # e.g. highest: [high, est</w>]
        self.merges = {}                          # e.g. [high, est</w>]: highest
        self.max_iter = max_iter
        self.debug = debug

    def train(self) -> None:
        # count word freq
        for document in self.corpus:
            words: list[str] = document.split()
            self.word_freq.update(words)

        # init self.splits
        for key in self.word_freq:                      # key is the word in counter word_freq
            self.splits[key] = list(key) + ["</w>"]     # e.g. highest: [high, est</w>]

        if self.debug:
            print(f"init splits: {self.splits}")

        alphabet = set()
        for key in self.word_freq:
            alphabet |= set(list(key))
        alphabet.add("</w>")

        self.vocab = list(alphabet)
        self.vocab.sort()
        # print(self.vocab)

        cnt = 0
        while len(self.vocab) < self.vocab_size:
            if self.max_iter and cnt > self.max_iter:
                break

            pair_freq = self.get_pair_freq()            # get pairs(i, i + 1) with their freqs
            if len(pair_freq) == 0:
                print("No pair available")
                break

            pair = max(pair_freq, key=pair_freq.get)    # get the most freq pair(i, i + 1)

            self.update_splits(pair[0], pair[1])

            if self.debug:
                print(f"update splits: {self.splits}")

            self.merges[pair] = pair[0] + pair[1]
            self.vocab.append(pair[0] + pair[1])

            if self.debug:
                print(
                    f"Most freq pair({max(pair_freq.values())} times) "
                    f"is : {pair[0]}, {pair[1]}. Vocab size: {len(self.vocab)}"
                )

            cnt += 1

    def get_pair_freq(self) -> dict:
        pairs_freq = defaultdict(int)
        for word, freq in self.word_freq.items():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pairs_freq[(split[i], split[i + 1])] += freq

        return pairs_freq

    def update_splits(self, s1: str, s2: str) -> None:
        for word, splits in self.splits.items():
            new_splits = []
            cursor = 0
            while cursor < len(splits):
                if cursor + 1 < len(splits) and (splits[cursor], splits[cursor + 1]) == (s1, s2):
                    new_splits.append(s1 + s2)
                    cursor += 2
                else:
                    new_splits.append(splits[cursor])
                    cursor += 1
            self.splits[word] = new_splits


    def tokenize(self, s: str) -> list[str]:
        splits: list[list[str]] = [list(w) + ["</w>"] for w in s.split()]

        for s1, s2 in self.merges:
            for i, split in enumerate(splits):
                new_split = []
                cursor = 0
                while cursor < len(split):
                    if cursor + 1 < len(split) and (split[cursor], split[cursor + 1]) == (s1, s2):
                        new_split.append(s1 + s2)
                        cursor += 2
                    else:
                        new_split.append(split[cursor])
                        cursor += 1
                assert "".join(new_split) == "".join(split)
                splits[i] = new_split
        return sum(splits, [])


if __name__ == "__main__":
    corpus = ["highest", "higher", "lower", "lowest", "cooler", "coolest"]
    bpe = BPE(corpus=corpus, vocab_size=25)
    bpe.train()
    print(bpe.tokenize(" ".join(corpus)))
    print(bpe.vocab)
    print(len(bpe.vocab))
    print(bpe.splits)
    print(bpe.merges)
