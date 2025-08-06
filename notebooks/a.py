import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class TreeNode:
    def __init__(self, is_leaf=False, class_label=None, svm_model=None):
        self.is_leaf = is_leaf          # True/False
        self.class_label = class_label   # Only for leaves
        self.svm_model = svm_model       # Trained SVC for non-leaves
        self.left = None                 # "Below" boundary child
        self.right = None                # "Above/on" boundary child

def build_tree(X, y, depth=0, max_depth=5, kernel='linear'):
    # Stopping conditions
    if (depth >= max_depth) or (len(X) < 10) or (len(np.unique(y)) == 1):
        majority_class = np.argmax(np.bincount(y))
        return TreeNode(is_leaf=True, class_label=majority_class)
    
    best_gini = float('inf')
    best_svm = None
    best_mask = None
    
    # Try all 1-class and 2-class combinations
    classes = np.unique(y)
    for class_combo in generate_combinations(classes):  # Implement this helper
        # Create binary labels: 1=selected class(es), 0=others
        y_binary = np.isin(y, class_combo).astype(int)
        
        # Train SVM
        svm = SVC(kernel=kernel).fit(X, y_binary)
        
        # Evaluate split using Gini index
        mask = svm.predict(X) == 1
        gini = calculate_gini(y[mask], y[~mask])  # Implement this
        
        if gini < best_gini:
            best_gini, best_svm, best_mask = gini, svm, mask
    
    # Recursively build subtrees
    node = TreeNode(svm_model=best_svm)
    node.left = build_tree(X[~best_mask], y[~best_mask], depth+1, max_depth, kernel)
    node.right = build_tree(X[best_mask], y[best_mask], depth+1, max_depth, kernel)
    return node

def generate_combinations(classes):
    combos = []
    for c in classes:
        combos.append([c])
    for i in range(len(classes))

def predict(node, x):
    while not node.is_leaf:
        side = node.svm_model.predict([x])[0]  # 0=left, 1=right
        node = node.right if side == 1 else node.left
    return node.class_label