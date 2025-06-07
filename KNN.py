import numpy as np
from collections import Counter

class KNN_featureselection:
    def __init__(self, data, k=1):
        self.k = k # User can input a K
        self.data = self.scale(data) # Always scale the features

    def scale(self, data):
        labels = data[:, 0].reshape(-1, 1)
        features = data[:, 1:]
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds[stds == 0] = 1
        scaled = (features - means) / stds #Scale features to mean0 and variance 1
        return np.hstack((labels, scaled))

    #def euclidean_distance(self, x, y): #too slow
    #    return np.sqrt(np.sum((x - y) ** 2))

    def nofeature(self):
        labels, counts = np.unique(self.data[:, 0], return_counts=True) #default rate
        return np.max(counts) / len(self.data)

    def leave_one_out_knn(self, data, feature_indices):
        X = data[:, feature_indices].astype(np.float32) #features
        y = data[:, 0] #labels
        n = len(data)
        correct = 0
    
        for i in range(n):
            test_x = X[i] #leave one out as the test instance
            test_y = y[i]
    
            train_x = np.delete(X, i, axis=0) #other instance are training data
            train_y = np.delete(y, i, axis=0)
              
            distances = np.linalg.norm(train_x - test_x, axis=1)  #compute distances in vectorized way
            
            k_indices = np.argpartition(distances, self.k)[:self.k]  #indices of the k smallest distances
            k_labels = train_y[k_indices]
    
            predicted = Counter(k_labels).most_common(1)[0][0]   #majority vote
     
            if predicted == test_y:
                correct += 1
    
        return correct / n

        
def forward_selection(self):
    total_features = self.data.shape[1] - 1  # total number of features
    current_features = []  # currently selected features
    best_overall_features = []  # best feature subset so far
    best_overall_accuracy = 0.0  # best accuracy so far

    base_accuracy = self.nofeature()  # default rate
    history = [(0, [], base_accuracy)]
    print("\nForward selection.")
    print(f" No features with {base_accuracy:.4f}")

    for i in range(total_features): 
        feature_to_add = None
        best_accuracy_now = 0.0  # best accuracy in this round

        for candidate in range(1, total_features + 1):  # try adding each unselected feature
            if candidate not in current_features:
                temp_features = current_features + [candidate]
                accuracy = self.leave_one_out_knn(self.data, temp_features)
                print(f" Test features {temp_features}, leave-one-out CV accuracy = {accuracy:.4f}")

                if accuracy > best_accuracy_now:  # update if this candidate gives better accuracy
                    best_accuracy_now = accuracy
                    feature_to_add = candidate

        if feature_to_add is not None:
            current_features.append(feature_to_add)  # ddd the best new feature 
            print(f"Add feature {feature_to_add} to {current_features} with accuracy: {best_accuracy_now:.4f}")

            if best_accuracy_now > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_now
                best_overall_features = current_features.copy()

        history.append((i + 1, current_features.copy(), best_accuracy_now, feature_to_add)) #save trace in history

    print(f" Best feature subset: {best_overall_features} with accuracy: {best_overall_accuracy:.4f}")
    return best_overall_features, best_overall_accuracy, history


def backward_selection(self):
    total_features = self.data.shape[1] - 1
    current_features = list(range(1, total_features + 1))  #start with all features

    history = []
    print("\nBackward elimination.")

    best_overall_accuracy = self.leave_one_out_knn(self.data, current_features)
    best_overall_features = current_features.copy()

    print(f" Initial full feature set {current_features}, leave-one-out accuracy = {best_overall_accuracy:.4f}")
    history.append((1, current_features.copy(), best_overall_accuracy))

    step = 1
    while len(current_features) > 1:  
        feature_to_remove = None
        best_accuracy_now = 0.0

        for candidate in current_features:  #  try removing each feature
            temp_features = current_features.copy()
            temp_features.remove(candidate)
            accuracy = self.leave_one_out_knn(self.data, temp_features)
            print(f" Test removing {candidate} from {temp_features}, leave-one-out CV accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy_now:  # update if removal improves accuracy
                best_accuracy_now = accuracy
                feature_to_remove = candidate

        if feature_to_remove is not None:
            current_features.remove(feature_to_remove)  # remove the worst feature
            print(f"Remove feature {feature_to_remove} from {current_features} with accuracy: {best_accuracy_now:.4f}")

            if best_accuracy_now > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_now
                best_overall_features = current_features.copy()

            step += 1
           
            history.append((step, current_features.copy(), best_accuracy_now, feature_to_remove)) #save trace in history
        else:
            print("No feature removal improved accuracy.")
            break

    base_accuracy = self.nofeature()
    history.append((step + 1, [], base_accuracy))
    print(f" No features with accuracy: {base_accuracy:.4f}")

    print(f"Best feature subset: {best_overall_features} with accuracy: {best_overall_accuracy:.4f}")
    return best_overall_features, best_overall_accuracy, history




large = np.loadtxt('/rhome/ymu015/bigdata/cs205/CS205_large_Data__20.txt')


selector = KNN_featureselection(large, k=5) #define selector

import time
start_forward = time.time()
forward_features, forward_acc, forward_hist = selector.forward_selection() 
end_forward = time.time()
print(f"Forward selection took {end_forward - start_forward:.2f} seconds.")
with open('/rhome/ymu015/bigdata/cs205/forward_history_k5.txt', 'w') as f:
    for item in forward_hist:
        f.write(f"{item}\n")
        
start_backward = time.time()
backward_features, backward_acc, backward_hist = selector.backward_selection()
end_backward = time.time()
print(f"Backward selection took {end_backward - start_backward:.2f} seconds.")

with open('/rhome/ymu015/bigdata/cs205/backward_history_k5.txt', 'w') as f:
    for item in backward_hist:
        f.write(f"{item}\n")


