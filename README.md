# 第一屆 NLP 經典機器學習馬拉松

## Day 1 ~ 2 : Python 文字處理函數介紹

## Day 3 ~ 4 : 正規表達式

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
* 找出所有中文字 & 數字的方法：`re.findall(r'[\u4E00-\u9FA50-9]'`
* `namedtuple(Name, attr)` 的使用
* `sorted(iter_obj, key=lambda x: x[index])` 代表設定 x[index] 為排序依據

## Day 10 ~ 11: 詞性標註 (POS Tagging)
* 決定單詞的詞性前，除了考慮單詞本身，也要考慮前後單詞與整句話
* 通常一個單詞會包含多種詞性
* `詞幹提取(stemming)`：去除詞綴得到詞根
* `Closed Class`：相對固定的詞類，不太會有新的詞類出現
    * `pronouns`: she, he, I
    * `preposition`: on, under, by
* `Open Class`：容易有新詞被創造
    * 如 `Noun`、`Verb`、`Adjective`等等
* `jieba.cut()`回傳一個 Generator，記得使用 `join` 來 `print`

## Day 12 : 詞袋模型 (Bag-of-words)
* 步驟
    1. 資料集包含正反面評價 1000 則
    2. 所有的單詞建一個字典 (每個單詞有對應的 index，不限順序，但不可改變)
    3. 假設字典大小為 3000 (也就是 3000 個單詞)，每則評價視為一袋，要用一個向量表示這個評價
    5. 先建一個 3000 維皆為 0 的向量 (ex.[0, 0, 0,......])，再將這個評價內有出現的單詞取出，找到對應的 index，將向量中這個位置的值 +1
    6. 若一個評價中找到兩個 good，而 good 對應到的 index 為 5，所以我們就在向量 [5] 的位置 +2，變為 [0, 0, 0, 0 , 0, 2,.....]
* 優點
    * 直觀，操作容易，並且不需要任何預訓練模型，可套用在任何需要將文字轉向量的任務上
* 缺點
    * 無法表達前後語意關係
    * 無法呈現單字含義：許多單字有多種不同含義，如我要買蘋果手機跟我要去菜市場買蘋果，兩句話中的蘋果意義不相同，但在 Bag-of-words 中無法呈現。
    * 形成稀疏矩陣，不利於部分模型訓練：假設我們訓練的 corpus 內有 100000 個單字，那要表達每一個單字就是(1,100000) 的向量，其中絕大部分都是 0 的數值。



## Day 13 : 詞幹/詞條提取
* 優點
    * 降低單詞數量，避免向量過於稀疏
    * 降低複雜度，加快模型訓練速度
* 缺點
    * 失去部分訊息 (e.g. ing 時態訊息被刪掉)
* `Stemming`: 依照規則刪除字尾
* `Lemmatization`: 取出單詞的 Lemma (詞條、詞元 = 字的原型)
* SOTA model 指在特定研究任務或 benchmark（基準） 資料集上，目前最突出的 model。
* 現今 SOTA 模型中較少用到 Stemming / Lemmatization 的技術，取而代之的是運用 NLP 模型 (e.g. BERT) 來進行單詞拆解，常見如 `Wordpiece`

## Day 14 : 文字預處理
### 預處理順序整理
* 匯入套件
* 讀取資料
* 去除部分字元、轉小寫 `re.sub()`
* 斷詞斷句：英文用 `nltk.word_tokenize()`、中文用 `jieba.cut()`
* 移除贅字：`nltk.download('stopwords')`
* 詞幹提取（英文）`PorterStemmer()`

### 預測
* 轉為詞袋：`CountVectorizer()`
* 訓練預測分組：`train_test_split()`
* 訓練：`classifier.fit()`
* 預測：`classifier.predict()`

## Day 15 : TF-IDF
* 詞頻（term frequency，TF）指的是某一個給定的詞語在該檔案中出現的頻率
* 逆向檔案頻率（inverse document frequency，IDF）是詞語普遍重要性的度量，由**總檔案數目**除以**包含該詞語之檔案的數目**，再將得到的**商取以10為底的對數**得到
* 字詞的重要性隨著它在檔案中出現的次數成正比增加，但同時會隨著它在語料庫中出現的頻率成反比下降
