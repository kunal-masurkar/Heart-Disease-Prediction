import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#Node class
class Node:
    def __init__(self,left=None,right=None,threshold=None,feature=None,*,value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
   
#Tree class
class DecisionTree:
    def __init__(self,max_depth=10):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self,X,y):
        self.root = self._create_tree(X,y)
        
    def _create_tree(self, X, y, depth=0):
        sample_count, feature_count = X.shape
        label_count = len(np.unique(y))
        
        if label_count == 1 or depth>=self.max_depth:
            leaf_value = self._most_common(y)
            return Node(value = leaf_value)
        
        best_feature, threshold = self._best_split(X, y, feature_count)
        left_idxs = np.where(X[:, best_feature] <= threshold)[0]
        right_idxs = np.where(X[:, best_feature] > threshold)[0]
        left = self._create_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._create_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(left, right, threshold, best_feature)
        
        
        
    def _best_split(self,X,y,feature_count):
        best_info_gain = -1
        split_idx, split_threshold = None, None
        
        for feature_idx in range(feature_count):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                info_gain = self._cal_info_gain(X_column,y,threshold)
                if(info_gain > best_info_gain):
                    best_info_gain = info_gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx,split_threshold
                    
                                       
    def _cal_info_gain(self,X_column,y,threshold):
        parent_entropy = self._entropy(y)
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        left_size, right_size = left_idxs.size, right_idxs.size
        n = len(y)
        left_entropy, right_entropy = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        weighted_child_entropy = (left_size/n)*left_entropy + (right_size/n)*right_entropy
        info_gain = parent_entropy - weighted_child_entropy
        return info_gain
    
    def _entropy(self,y):
        unique_y = np.unique(y)
        cal = []
        for i in unique_y:
            p_num = (np.where(y == i))[0].size
            p_den = len(y)
            p = p_num / p_den
            cal.append(p*np.log2(p))
        return -np.sum(cal)
    
    def _most_common(self, y):
        most_common_val = Counter(y).most_common(1)[0][0]
        return most_common_val
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if(x[node.feature] > node.threshold):
            return self._traverse_tree(x, node.right)
        return self._traverse_tree(x, node.left)


if __name__ == "__main__":
    data = pd.read_csv("/Users/kunal/Desktop/DecisionTree/heart/heart.csv")
    X, y = data.values[:, 0:13], data.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(X_test[2],y_test[2])
    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    acc = accuracy(y_test, predictions)
    print("Accuracy of model: ",acc*100)
    print("\nSome Values predicted by model:")
    results = np.c_[y_test, predictions]
    compared_data = pd.DataFrame(results, columns=["Actual Value", "Predicted Value"])
    print(compared_data.head(10))
   

