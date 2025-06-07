import numpy as np
from collections import Counter
import pandas as pd
import time

class KNN_featureselection:
    def __init__(self, data, k=1):
        self.k = k
        self.data = self.scale(data)

    def scale(self, data):
        labels = data[:, 0].reshape(-1, 1)
        features = data[:, 1:]
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds[stds == 0] = 1
        scaled = (features - means) / stds
        return np.hstack((labels, scaled))

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def nofeature(self):
        labels, counts = np.unique(self.data[:, 0], return_counts=True)
        return np.max(counts) / len(self.data)

    def leave_one_out_knn(self, data, feature_indices):
        accurate = 0
        n = len(data)
    
        for i in range(n):
            test_data = data[i]
            test_features = test_data[feature_indices]
            test_label = test_data[0]
    
            distances = []
    
            for j in range(n):
                if i == j:
                    continue
                train_features = data[j][feature_indices]
                train_label = data[j][0]
    
                distance = self.euclidean_distance(test_features, train_features)
                distances.append((distance, train_label))
    
            #sort the distances
            distances.sort(key=lambda x: x[0])
    
            #get top k neighbors
            k_labels = [label for _, label in distances[:self.k]]
    
            #get the class
            predicted_label = Counter(k_labels).most_common(1)[0][0]
    
            if test_label == predicted_label:
                accurate += 1
    
        return accurate / n
        
    def forward_selection(self):
        total_features = self.data.shape[1] - 1  # exclude label
        current_features = []
        best_overall_features = []
        best_overall_accuracy = 0.0

        base_accuracy = self.nofeature()
        history = [(0, [], base_accuracy)]
        print("\nForward selection.")
        print(f" No features with {base_accuracy:.4f}")

        for i in range(total_features):
            feature_to_add = None
            best_accuracy_now = 0.0  

            for candidate in range(1, total_features + 1):
                if candidate not in current_features:
                    temp_features = current_features + [candidate]
                    accuracy = self.leave_one_out_knn(self.data, temp_features)
                    print(f" Test features {temp_features}, leave-one-out CV accuracy = {accuracy:.4f}")

                    if accuracy > best_accuracy_now:
                        best_accuracy_now = accuracy
                        feature_to_add = candidate

            if feature_to_add is not None:
                current_features.append(feature_to_add)
                print(f"Add feature {feature_to_add} to {current_features} with accuracy: {best_accuracy_now:.4f}")

                if best_accuracy_now > best_overall_accuracy:
                    best_overall_accuracy = best_accuracy_now
                    best_overall_features = current_features.copy()

            history.append((i + 1, current_features.copy(), best_accuracy_now, feature_to_add))

        print(f" Best feature subset: {best_overall_features} with accuracy: {best_overall_accuracy:.4f}")
        return best_overall_features, best_overall_accuracy, history

    def backward_selection(self):
        total_features = self.data.shape[1] - 1
        current_features = list(range(1, total_features + 1))

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

            for candidate in current_features:
                temp_features = current_features.copy()
                temp_features.remove(candidate)
                accuracy = self.leave_one_out_knn(self.data, temp_features)
                print(f" Test removing {candidate} from {temp_features}, leave-one-out CV accuracy = {accuracy:.4f}")

                if accuracy > best_accuracy_now:
                    best_accuracy_now = accuracy
                    feature_to_remove = candidate

            if feature_to_remove is not None:
                current_features.remove(feature_to_remove)
                print(f"Remove feature {feature_to_remove} from {current_features} with accuracy: {best_accuracy_now:.4f}")

                if best_accuracy_now > best_overall_accuracy:
                    best_overall_accuracy = best_accuracy_now
                    best_overall_features = current_features.copy()

                step += 1
                history.append((step, current_features.copy(), best_accuracy_now, feature_to_remove))
            else:
                print("No feature removal improved accuracy.")
                break

        base_accuracy = self.nofeature()
        history.append((step + 1, [], base_accuracy))
        print(f" No features with accuracy: {base_accuracy:.4f}")

        print(f"Best feature subset: {best_overall_features} with accuracy: {best_overall_accuracy:.4f}")
        return best_overall_features, best_overall_accuracy, history



df0 = pd.read_csv('weather_classification_data.csv')

df = np.array(df0)
season = df[:,7].copy() #season is the class label
temperature = df[:,0].copy()
df[:,7] = temperature
df[:,0] = season
data = df[:, [0,1,2,3,5,7,8]] #make season as the first column

# Encode season
for i in range(len(season)):
    if season[i] == 'Spring':
        data[i,0] = 1
    elif season[i] == 'Summer':
        data[i,0] = 2
    elif season[i] == 'Autumn':
        data[i,0] = 3
    elif season[i] == 'Winter':
        data[i,0] = 4

data = data.astype(float)

#Run KNN
selector = KNN_featureselection(data, k=1) #define selector


start_forward = time.time()
forward_features, forward_acc, forward_hist = selector.forward_selection() 
end_forward = time.time()
print(f"Forward selection took {end_forward - start_forward:.2f} seconds.")

with open('/rhome/ymu015/bigdata/cs205/results/Weather_forward.txt', 'w') as f:
    for item in forward_hist:
        f.write(f"{item}\n")

start_backward = time.time()
backward_features, backward_acc, backward_hist = selector.backward_selection()
end_backward = time.time()
print(f"Backward selection took {end_backward - start_backward:.2f} seconds.")

with open('/rhome/ymu015/bigdata/cs205/results/Weather_backward.txt', 'w') as f:
    for item in backward_hist:
        f.write(f"{item}\n")

