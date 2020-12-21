# 第一屆 NLP 經典機器學習馬拉松

## Day 1 : Python 文字處理函數介紹 - 1

## Day 2 : Python 文字處理函數介紹 - 2

## Day 3 : 正規表達式 - 基礎

## Day 4 : 正規表達式 - 進階

## Day 5 : NLP 中文斷詞概念
* Trie 樹
* 隱馬可夫模型 HMM
* 維特比動態規劃演算法 Viterbi

## Day 6 : 使用 Jieba 進行中文斷詞
* `set_dictionary()`
* `load_userdict()`
* `add_word()` 動態加入字典
* `suggest_freq()` 動態調整詞頻
* `posseg.cut()` 詞性標註
* `tokenize()` 斷詞位置標註

## Day 7 : 使用 CkipTagger 進行繁體中文斷詞
* 安裝 ckiptagger, tensorflow, gdown (Google Drive 下載)
* `WS` 斷詞
* `POS` 詞性標註
* `NER` 實體辨識
* `construct_dictionary(dict_word2weight)`
* `recommend_dictionary` 使用建議字典
* `coerce_dictionary` 使用強制字典（優先）

## Day 8 : 基礎語言模型：N-Gram
* Bigram 模型：`P(word1|start) * P(word2|word1) ... * P(end|word_x)`
    
## Day 9 : 基礎語言模型：N-Gram
* 以 Python 建立 N-Gram 模型
* 以 NLTK 套件實作 N-Gram
* 找出所有中文字的方法：`re.findall(r'[\u4E00-\u9FA50-9]'`
* `namedtuple(Name, attr)` 的使用
* `sorted(iter_obj, key=lambda x: x[index])` 代表設定 x[index] 為排序依據

## Day 10 : 詞性標註 (POS Tagging)
* 決定單詞的詞性前，除了考慮單詞本身，也要考慮前後單詞與整句話
* 通常一個單詞會包含多種詞性
* `詞幹提取(stemming)`：去除詞綴得到詞根
* `Closed Class`：相對固定的詞類，不太會有新的詞類出現
    * `pronouns`: she, he, I
    * `preposition`: on, under, by

* `Open Class`：容易有新詞被創造
    * 如 `Noun`、`Verb`、`Adjective`等等

## Day 11 : 詞性標註 (POS Tagging)
