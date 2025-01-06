#자동미분 using tensorflow
import tensorflow as tf

w = tf.Variable(2.)

def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:
    z = f(w)
gradients = tape.gradient(z, [w])
print(gradients)

w = tf.Variable(4.0)
b = tf.Variable(1.0)

@tf.function
def hypothesis(x):
    return w*x + b

@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

x = [1,2,3,4,5,6,7,8,9]
y = [11,12,13,14,15,16,17,77,89,90]
optimizer = tf.keras.optimizers.SGD(0.01)

for i in range(301):
    with tf.GradientTape() as tape:
        y_pred = hypothesis(x)
        cost = mse_loss(y_pred, y)

    gradients = tape.gradient(cost, [w,b])
    optimizer.apply_gradients(zip(gradients, [w,b]))
    if i % 10 == 0:
        print("epoch : {:3} | w : {:5.4f} | b : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))
        

#sigmoid function
import numpy as np
import matplotlib as plt
def sigmoid(x):
    return 1/(1+ np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, "g")
plt.plot([0,0], [1.0, 0.0], ":") #가운데 점선을 추가
plt.title("sigmoid_function")
plt.show()

import random
p = random(0, 1)
loss = -np.sum(y*np.log(p) + (1-y)*np.log(1-p))
#cross-entropy(실제 분포와 예측확률분포의 차이를 계산하는 함수)


