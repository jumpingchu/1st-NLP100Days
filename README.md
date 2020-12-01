# 第一屆 NLP 經典機器學習馬拉松

## Day 1
* Python 文字處理函數介紹 - 1

## Day 2
* Python 文字處理函數介紹 - 2

## Day 3
* 正規表達式 - 基礎

## Day 4
* 正規表達式 - 進階

## Day 5
* NLP 中文斷詞概念
    * Trie 樹
    * 隱馬可夫模型 HMM
    * 維特比動態規劃演算法 Viterbi

## Day 6
* 使用 Jieba 進行中文斷詞
    * `set_dictionary()`
    * `load_userdict()`
    * `add_word()` 動態加入字典
    * `suggest_freq()` 動態調整詞頻
    * `posseg.cut()` 詞性標註
    * `tokenize()` 斷詞位置標註

## Day 7
* 使用 CkipTagger 進行繁體中文斷詞
    * 安裝 ckiptagger, tensorflow, gdown (Google Drive 下載)
    * `WS` 斷詞
    * `POS` 詞性標註
    * `NER` 實體辨識
    * `construct_dictionary(dict_word2weight)`
    * `recommend_dictionary` 使用建議字典
    * `coerce_dictionary` 使用強制字典（優先）

## Day 8
* 基礎語言模型：N-Gram
    * Bigram 模型：`P(word1|start) * P(word2|word1) ... * P(end|word_x)`
    