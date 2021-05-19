import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

# 分割測試集與訓練集  
def train_test_split(df, test_size=0.1):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    #以隨機的方式取的測試集資料點的index
    indices = list(df.index)
    test_indices = random.sample(population=indices, k=test_size)

    #分割測試集與訓練集
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# 檢查資料是否都為同一類別
def check_purity(data):
    '''Function to check if input data all belong to the same class
    Parameter
    ---------
    data: list
        Input data
    '''
    #取的資料的label訊息
    labels = data[:, -1]
    
    #檢查是否所有的label都為同一種
    unique_classes = np.unique(labels)
    
    if len(unique_classes) == 1:
        return True
    else:
        return False
    
# 根據給定的資料，取得每個特徵(feature)可能做為樹型模型分割節點的值
# 可能作為分割節點得值即為每個特徵的獨特值(unique value)
def get_potential_splits(data, random_features=None):
    '''Function to get all potential split value for tree base model
    Parameter
    ---------
    data: list
        Input data
    '''
    
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1)) #此處的-1是為了扣掉label的欄位
    
    if random_features and random_features <= len(column_indices):
        #隨機選取特徵進行訓練
        column_indices = random.sample(population=column_indices, k=random_features)
    
    for column_index in column_indices:    
        
        #根據欄位取的特徵的獨特值(unique values)
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        #將取得的可能分割值除存在potential_split的字典中(key=特徵欄位的index, value:此特徵可能的分割值)
        potential_splits[column_index] = unique_values
    
    return potential_splits


#由給定的輸入DataFrame給個特徵值的型態(數值型特徵或類別型特徵)
def determine_type_of_feature(df):
    '''Function to get features types
    Parameter
    ---------
    df: pd.DataFrame
        Input raw pd.DataFrame data
    '''
    
    feature_types = []
    
    #若特徵的獨特值個數較少，及當作類別型特徵資料(若為數值型，獨特值個數應該會很多)
    #此處簡易的將判斷方法設為資料個數的1/3次方，此值可以自行修改選較為適合的個數
    n_unique_values_treshold = int(len(df)**(1/3))
    
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            rep_value = unique_values[0] #選出一個值做此特徵的代表

            if (isinstance(rep_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# 根據給定的資料、欲採用特徵欄位指標(index)與欲採用的分割值，來取的分割節點分割後的左節點資料與右節點資料
def split_data(data, split_column, split_value):
    '''Function to splitted left and right nodes
    Parameter
    ---------
    data: list
        Input data
    split_column: int
        index for feature column
    split_value: float or int or string
        value to be used as split benchmark
    '''
    
    #取得用來分割的特徵欄位
    split_column_values = data[:, split_column]

    #依據欄位值的型態(數值型特徵或類別型特徵)來進行節點分割
    type_of_feature = FEATURE_TYPES[split_column]
    
    if type_of_feature == "continuous":
        #數值型特徵分割
        data_left = data[split_column_values <= split_value]
        data_right = data[split_column_values >  split_value]
    else:
        #類別型特徵分割
        data_left = data[split_column_values == split_value]
        data_right = data[split_column_values != split_value]
    
    return data_left, data_right


# 根據給定的資料與任務類型(回歸或分類)來產生終端節點
def create_leaf(data, task_type):
    '''Function to create leaf node
    Parameters
    ----------
    data: list
        Input data
    task_type: str
        indicate the type of tree (regression or classification)
    '''
    
    #取的資料的label欄位
    label_column = data[:, -1]
    
    if task_type == "regression":
        #回歸任務
        leaf = np.mean(label_column)
    else:
        #分類任務
        #取得所有輸入資料的獨立類別與其個數
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        
        #以個數最多的類別，作為此節點的輸出類別
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    
    return leaf

#計算資料的熵(Entropy)
def calculate_entropy(data):
    
    #取的資料的label訊息
    label_column = data[:, -1]
    
    #取得所有輸入資料的獨立類別與其個數
    _, counts = np.unique(label_column, return_counts=True)

    #計算機率
    probabilities = counts / counts.sum()
    
    #計算entropy
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


#取得左節點與右節點訊息合
def calculate_overall_metric(data_below, data_above, metric_function):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric =  (p_data_below * metric_function(data_below) 
                     + p_data_above * metric_function(data_above))
    
    return overall_metric


#以迴圈的方式計算所有可能分割值的訊息增益，取的最佳的分割特徵與值(訊息增益最大)
def determine_best_split(data, potential_splits, metric_function, task_type='classification'):
    
    #紀錄是否為樹的第一層(第一次回圈)
    first_iteration = True
    
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            
            #根據給定的特徵與分割值分割資料為左節點、右節點
            data_left, data_right = split_data(data, split_column=column_index, split_value=value)
            
            #判斷是回歸樹亦或分類樹
            if task_type == "regression":
                #回歸樹
                current_overall_metric = calculate_overall_metric(data_left, data_right, metric_function=metric_function)
            else:
                #分類樹
                current_overall_metric = calculate_overall_metric(data_left, data_right, metric_function=metric_function)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


class decision_tree():
    '''Decision Tree model
    Parameters
    -----------
    metric_function: function
        the metric function used to calculate information gain
    task_type: str
        indicate the type of tree (regression or classification)
    counter: int
        counter for recording number of splits
    min_samples: int
        minimum number of samples for a node to be able to split
    max_depth: int
        Maximum depth for the decision tree
    '''
    def __init__(self, metric_function, task_type='classification', counter=0, min_samples=2, max_depth=5, random_features=None):
        
        self.metric_function = metric_function
        self.task_type = task_type
        self.counter = counter
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.random_features = random_features
    
    def fit(self, df):
        '''
        df: pd.DataFrame
            input raw DataFrame data
        '''
        # 資料準備
        if self.counter == 0:
            #若為第一次分割，取出資料特徵的欄位與其對應的型態
            global COLUMN_HEADERS, FEATURE_TYPES

            #取得資料特徵欄位
            COLUMN_HEADERS = df.columns
            #取的特徵型態
            FEATURE_TYPES = determine_type_of_feature(df)
            #取得資料特徵值
            data = df.values
        else:
            #取得資料特徵值
            data = df           

        # 終端節點處理(leaf)
        # 若資料都屬於同一種類別、資料個數小於最小可分割個數、樹的深度大於最大深度，節點即屬於終端節點(leaf)
        if (check_purity(data)) or (len(data) < self.min_samples) or (self.counter == self.max_depth):
            leaf = create_leaf(data, self.task_type)
            return leaf

        # 分割節點
        else:    
            self.counter += 1

            # 節點分割的左節點與右節點
            potential_splits = get_potential_splits(data, self.random_features)
            split_column, split_value = determine_best_split(data, potential_splits,
                                                             self.metric_function, self.task_type)
            data_left, data_right = split_data(data, split_column, split_value)

            # 若分割後的左節點或右節點sample個數為零(代表母節點即無法在分割)
            if len(data_left) == 0 or len(data_right) == 0:
                # 取出此節點
                leaf = create_leaf(data, self.task_type)
                return leaf

            # 取得分割節點的分割依據(特徵與分切值)
            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]

            if type_of_feature == "continuous":
                #連續型數值
                question = "{} <= {}".format(feature_name, split_value)
            else:
                #類別型數值
                question = "{} = {}".format(feature_name, split_value)

            # 建構子樹(sub-tree)
            sub_tree = {question: []}

            # 已遞迴的方式取建構完整決策樹    
            yes_answer = self.fit(data_left)
            no_answer = self.fit(data_right)

            #若左節點與右節點分割的結果相同，則此節點及不需再進行分割
            #此情形會發生在此節點資料個數小於min_samples或樹深度大於max_depth
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
            
            self.sub_tree = sub_tree
            
            return self.sub_tree
        
    def pred(self, example, tree):
        # 使用訓練好的決策樹進行預測
        
        #取得分割節點(由上到下)
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")

        #以節點分割問題分類資料
        if comparison_operator == "<=":
            #數值型資料
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        else:
            #類別型資料
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        
        # 若分類完成，返回分類結果
        if not isinstance(answer, dict):
            return answer
        else:
            #繼續往下分類
            residual_tree = answer
            return self.pred(example, residual_tree)