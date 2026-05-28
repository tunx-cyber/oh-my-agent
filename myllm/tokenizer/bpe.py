from collections import Counter, defaultdict
import re
class BPETokenizer:

    # ══════════════════════════════════════════════════
    #  预分词正则
    # ══════════════════════════════════════════════════

    # GPT-2 风格: 按「词 + 前导空格」「标点」「空白」分别切分
    #   \s+(?!\S)  → 跟在非空白后面的空白序列（即词尾空格归入下一个 chunk）
    #   \s+        → 其余空白（连续空白、独立空白行等）
    GPT2_PRETOKENIZE = re.compile(
        r"""
          's|'t|'re|'ve|'m|'ll|'d     # 英文缩写
        |[ ]?[a-zA-Z]+                 # 字母词（可带前导空格）
        |[ ]?[0-9]+                    # 数字
        |[ ]?[^\s\w]+                  # 标点符号
        |\s+                           # 任意空白序列
        """,
        re.VERBOSE,
    )


    # 简单版：保留所有字符，空白单独成 token
    SIMPLE_PRETOKENIZE = re.compile(
        r"""
          \S+      # 非空白连续串
        | \s+      # ★ 空白连续串（\n \t \r 空格，一个都不少）
        """,
        re.VERBOSE,
    )

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self._pat = self.GPT2_PRETOKENIZE

    def _pretokenize(self, text: str) -> list[str]:
        """
        用正则把原始文本切成 chunk，每个 chunk 独立做 BPE。

        不再 .strip()，不再 .split()！

        示例:
          "hello\nworld\tfoo"
          → ['hello', '\n', 'world', '\t', 'foo']

          "  foo  bar\n\nbaz"
          → ['  ', 'foo', '  ', 'bar', '\n\n', 'baz']
        """
        return self._pat.findall(text)
    
    def _chunk_to_token_tuple(self, chunk: str) -> tuple[str, ...]:
        """
        把一个 chunk 拆成字符序列 + 词尾标记。

        空白 chunk 不加 </w>，因为它们本身就是分隔符。
        非空白 chunk 加 </w> 标记词尾。
        """
        chars = list(chunk)
        if chunk.strip():  # 非纯空白
            return tuple(chars + ["</w>"])
        else:              # 纯空白（\n, \t, 空格等）
            return tuple(chars)
        
    def _get_pair_stats(self, word_freqs:Counter):
        """统计所有相邻 token 对的频率"""
        paris = Counter()
        # tokens是原来每个单词后list化后不断融合的
        for tokens, freq in word_freqs.items():
            for i in range(len(tokens)-1):
                paris[(tokens[i],tokens[i+1])] += freq
        return paris
    
    def _merge_pair(self, pair, word_freqs:Counter):
        new_word_freqs = {}
        merged = pair[0]+pair[1]

        for tokens, freq in word_freqs.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 对于一个tuple里面的词，把能合并的都合并了，然后记录这个新的tuple
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
                
                new_word_freqs[tuple(new_tokens)] = freq
        
        return new_word_freqs
    
    def train(self, text: str, vocab_size:int):
        """
        训练 BPE 分词器

        Args:
            text:       训练语料字符串
            vocab_size: 目标词表大小（包含初始字符集）
        """

        # 将文本按空格切分为单词
        words = self._pretokenize(text)
        '''
        Counter 是 Python 标准库 collections 模块中的一个字典子类，
        专门用于统计可哈希对象（如列表、字符串中的元素）的出现次数。
        它以元素为键，出现次数为值，提供了方便的计数方法。
        '''
        word_freqs = Counter()
        
        # 每个单词拆成字符序列,
        for word in words:
            key = self._chunk_to_token_tuple(word)
            word_freqs[key] += 1
        
        base_vocab = set()
        for tokens in word_freqs:# 遍历key
            base_vocab.update(tokens)

        # 建立初始词表
        self.vocab = {token: idx for idx, token in enumerate(sorted(base_vocab))}

        # 迭代合并
        num_iters = vocab_size - len(self.vocab)

        for step in range(num_iters):
            pair_stats = self._get_pair_stats(word_freqs)
            if not pair_stats:
                break
            
            # max 会找出使得 pair_stats.get(k) 最大的那个键 k。best_pair是(x,y) tuple
            best_pair = max(pair_stats, key=pair_stats.get)

            # 记录合并规则
            merged_token = best_pair[0]+best_pair[1]
            self.merges[best_pair] = merged_token

            # 加入词表
            self.vocab[merged_token] = len(self.vocab)

            word_freqs = self._merge_pair(best_pair, word_freqs)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        return self
    
    def tokenize(self, text: str) -> list[str]:
        chunks = self._pretokenize(text)
        all_tokens = []

        for chunk in chunks:
            tokens = list(self._chunk_to_token_tuple(chunk))

            for pair, merged in self.merges.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1
                            and tokens[i] == pair[0]
                            and tokens[i + 1] == pair[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            all_tokens.extend(tokens)

        return all_tokens
    
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        unk = self.vocab.get("<UNK>", -1)
        return [self.vocab.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, "") for i in ids]
        text = ""
        for t in tokens:
            if t == "</w>":
                pass  # 词尾标记不输出，词间的空格由空白 token 本身提供
            else:
                text += t
        return text
    
if __name__ == "__main__":
    # 训练语料（故意用有大量重复模式的小语料来演示）
    corpus = (
        "hello world\n"
        "hello\tworld\n"
        "foo bar\n"
        "foo\tbar\n"
        "the quick brown fox\n"
        "the\tquick\tbrown\tfox\n"
    )

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=30)

    # 测试分词
    test_texts = ["hello world",
        "hello\tworld",
        "hello\nworld",
        "foo\tbar\n",
        "the quick brown fox",]

    print("\n" + "=" * 50)
    print("分词测试")
    print("=" * 50)

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"  '{text}'")
        print(f"    tokens : {tokens}")
        print(f"    ids    : {ids}")
        print(f"    decoded: '{decoded}'")
        print()    


