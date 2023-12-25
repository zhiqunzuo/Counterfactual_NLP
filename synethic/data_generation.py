import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

data_size = 999999

class generator(object):
    def __init__(self):
        self.sample_u()
        self.sample_a()
        self.sample_income()
        self.sample_property()
        self.sample_target()
    
    def sample_u(self):
        #self.u_1 = np.random.choice([-1, 0, 1], size=data_size)
        #self.u_2 = np.random.choice([-1, 0, 1], size=data_size)
        self.u_1 = np.array([1] * (data_size // 3) + [0] * (data_size // 3) + [-1] * (data_size // 3))
        self.u_2 = np.array([1] * (data_size // 3) + [0] * (data_size // 3) + [-1] * (data_size // 3))
        np.random.shuffle(self.u_1)
        np.random.shuffle(self.u_2)
    
    def sample_a(self):
        self.a = np.random.choice([-1, 1], size=data_size, p=[0.8, 0.2])
    
    def sample_income(self):
        self.i = self.u_1 + self.a
        
    def sample_property(self):
        self.p = self.u_2
    
    def sample_target(self):
        self.y = np.where((self.i + self.p) < 0, -1, 1)

data = generator()

input_features = np.zeros((data_size, 3))

for i in range(data_size):
    input_features[i, 1] = data.p[i]
    input_features[i, 2] = data.a[i]
    if data.a[i] == 1 and data.i[i] == 0:
        input_features[i, 0] = -1
    elif data.a[i] == 1 and data.i[i] == 1:
        input_features[i, 0] = 0
    elif data.a[i] == 1 and data.i[i] == 2:
        input_features[i, 0] = 1
    elif data.a[i] == -1 and data.i[i] == -2:
        input_features[i, 0] = -1
    elif data.a[i] == -1 and data.i[i] == -1:
        input_features[i, 0] = 0
    elif data.a[i] == -1 and data.i[i] == 0:
        input_features[i, 0] = 1

count_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
count_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(data_size):
    if input_features[i, 2] == 1:
        if input_features[i, 0] == -1 and input_features[i, 1] == -1:
            count_1[0] += 1
        elif input_features[i, 0] == -1 and input_features[i, 1] == 0:
            count_1[1] += 1
        elif input_features[i, 0] == -1 and input_features[i, 1] == 1:
            count_1[2] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == -1:
            count_1[3] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == 0:
            count_1[4] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == 1:
            count_1[5] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == -1:
            count_1[6] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == 0:
            count_1[7] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == 1:
            count_1[8] += 1
    else:
        if input_features[i, 0] == -1 and input_features[i, 1] == -1:
            count_2[0] += 1
        elif input_features[i, 0] == -1 and input_features[i, 1] == 0:
            count_2[1] += 1
        elif input_features[i, 0] == -1 and input_features[i, 1] == 1:
            count_2[2] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == -1:
            count_2[3] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == 0:
            count_2[4] += 1
        elif input_features[i, 0] == 0 and input_features[i, 1] == 1:
            count_2[5] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == -1:
            count_2[6] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == 0:
            count_2[7] += 1
        elif input_features[i, 0] == 1 and input_features[i, 1] == 1:
            count_2[8] += 1

print(np.array(count_1) / sum(count_1))
print(np.array(count_2) / sum(count_2))            

target = data.y


X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=42)

clf = SVC()

clf.fit(X_train[:, :2], y_train)

y_pred = clf.predict(X_test[:, :2])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

X_test_1 = X_test[X_test[:, 2] == 1, :2]
y_test_1 = y_test[X_test[:, 2] == 1]

y_pred_1 =clf.predict(X_test_1)
counts = np.count_nonzero(y_pred_1 == 1)
print("positive ratio = {}".format(counts / len(y_pred_1)))
#accuracy_1 = accuracy_score(y_test_1, y_pred_1)
#print("Group 1 accuracy = {}".format(accuracy_1))

X_test_2 = X_test[X_test[:, 2] == -1, :2]
y_test_2 = y_test[X_test[:, 2] == -1]

y_pred_2 =clf.predict(X_test_2)
counts = np.count_nonzero(y_pred_2 == 1)
print("positive ratio = {}".format(counts / len(y_pred_2)))
#accuracy_2 = accuracy_score(y_test_2, y_pred_2)
#print("Group 2 accuracy = {}".format(accuracy_2))