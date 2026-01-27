import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

class TextProcessor:
    def __init__(self):
        # 下载必要的NLTK资源
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        预处理文本
        :param text: 输入文本
        :return: 预处理后的文本
        """
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        # 转换为小写
        text = text.lower()
        return text
    
    def tokenize(self, text):
        """
        分词
        :param text: 输入文本
        :return: 分词结果
        """
        # 分句
        sentences = sent_tokenize(text)
        # 分词
        tokenized_sentences = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            # 去除停用词和标点符号
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            tokenized_sentences.append(filtered_tokens)
        return tokenized_sentences
    
    def extract_key_phrases(self, text):
        """
        提取关键短语
        :param text: 输入文本
        :return: 关键短语列表
        """
        # 预处理
        text = self.preprocess(text)
        # 分词
        tokenized_sentences = self.tokenize(text)
        
        # 提取关键短语（这里使用简单的n-gram方法）
        key_phrases = []
        for tokens in tokenized_sentences:
            # 提取2-gram和3-gram
            for i in range(len(tokens)):
                if i + 1 < len(tokens):
                    key_phrases.append(' '.join(tokens[i:i+2]))
                if i + 2 < len(tokens):
                    key_phrases.append(' '.join(tokens[i:i+3]))
        
        # 去重
        key_phrases = list(set(key_phrases))
        return key_phrases
    
    def process_input(self, text):
        """
        处理输入文本，返回预处理后的文本和关键短语
        :param text: 输入文本
        :return: 预处理后的文本和关键短语
        """
        # 预处理
        processed_text = self.preprocess(text)
        # 提取关键短语
        key_phrases = self.extract_key_phrases(processed_text)
        return processed_text, key_phrases
