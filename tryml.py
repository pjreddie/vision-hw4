from uwimg import *

def softmax_model():
    l = [make_layer(785, 10, SOFTMAX)]
    return make_model(l)

def neural_net():
    l = [   make_layer(785, 30, LOGISTIC),
            make_layer(30, 10, SOFTMAX)]
    return make_model(l)

print("loading data...")
train = load_classification_data("mnist.train", "mnist.labels", 1)
test  = load_classification_data("mnist.test", "mnist.labels", 1)
print("done")
print

print("training model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .0

m = neural_net()
train_model(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_model(m, train))
print("test accuracy:     %f", accuracy_model(m, test))

