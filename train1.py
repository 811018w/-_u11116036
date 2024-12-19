from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

# 載入 MNIST 資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 預處理資料
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255

# 建立模型
knn = KNeighborsClassifier(n_neighbors=5)
print('training...')
knn.fit(x_train, y_train)
print('ok')

# 儲存模型 (使用 joblib)
import joblib
joblib.dump(knn, 'mnist_knn.pkl')

# 測試模型
print('testing...')
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
