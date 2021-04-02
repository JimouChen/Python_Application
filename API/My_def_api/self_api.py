import matplotlib.pyplot as plt

string = '''
Train Epoch: 1 [14848/60000 (25%)]	Loss: 0.403663
Train Epoch: 1 [30208/60000 (50%)]	Loss: 0.238938
Train Epoch: 1 [45568/60000 (75%)]	Loss: 0.125005

Test set: Average loss: 0.0912, Accuracy: 9725/10000 (97%) 

Train Epoch: 2 [14848/60000 (25%)]	Loss: 0.098218
Train Epoch: 2 [30208/60000 (50%)]	Loss: 0.073512
Train Epoch: 2 [45568/60000 (75%)]	Loss: 0.091101

Test set: Average loss: 0.0572, Accuracy: 9826/10000 (98%) 

Train Epoch: 3 [14848/60000 (25%)]	Loss: 0.060899
Train Epoch: 3 [30208/60000 (50%)]	Loss: 0.041224
Train Epoch: 3 [45568/60000 (75%)]	Loss: 0.055987

Test set: Average loss: 0.0417, Accuracy: 9866/10000 (99%) 

Train Epoch: 4 [14848/60000 (25%)]	Loss: 0.051169
Train Epoch: 4 [30208/60000 (50%)]	Loss: 0.036964
Train Epoch: 4 [45568/60000 (75%)]	Loss: 0.025848

Test set: Average loss: 0.0389, Accuracy: 9869/10000 (99%) 

Train Epoch: 5 [14848/60000 (25%)]	Loss: 0.035927
Train Epoch: 5 [30208/60000 (50%)]	Loss: 0.012666
Train Epoch: 5 [45568/60000 (75%)]	Loss: 0.027482

Test set: Average loss: 0.0324, Accuracy: 9889/10000 (99%) 

Train Epoch: 6 [14848/60000 (25%)]	Loss: 0.011192
Train Epoch: 6 [30208/60000 (50%)]	Loss: 0.063323
Train Epoch: 6 [45568/60000 (75%)]	Loss: 0.013876

Test set: Average loss: 0.0345, Accuracy: 9879/10000 (99%) 

Train Epoch: 7 [14848/60000 (25%)]	Loss: 0.014208
Train Epoch: 7 [30208/60000 (50%)]	Loss: 0.010832
Train Epoch: 7 [45568/60000 (75%)]	Loss: 0.017518

Test set: Average loss: 0.0307, Accuracy: 9898/10000 (99%) 

Train Epoch: 8 [14848/60000 (25%)]	Loss: 0.018894
Train Epoch: 8 [30208/60000 (50%)]	Loss: 0.018114
Train Epoch: 8 [45568/60000 (75%)]	Loss: 0.012830

Test set: Average loss: 0.0342, Accuracy: 9883/10000 (99%) 

Train Epoch: 9 [14848/60000 (25%)]	Loss: 0.017432
Train Epoch: 9 [30208/60000 (50%)]	Loss: 0.017527
Train Epoch: 9 [45568/60000 (75%)]	Loss: 0.012198

Test set: Average loss: 0.0329, Accuracy: 9894/10000 (99%) 

Train Epoch: 10 [14848/60000 (25%)]	Loss: 0.009825
Train Epoch: 10 [30208/60000 (50%)]	Loss: 0.017382
Train Epoch: 10 [45568/60000 (75%)]	Loss: 0.008634

Test set: Average loss: 0.0329, Accuracy: 9897/10000 (99%) 

Train Epoch: 11 [14848/60000 (25%)]	Loss: 0.010014
Train Epoch: 11 [30208/60000 (50%)]	Loss: 0.004431
Train Epoch: 11 [45568/60000 (75%)]	Loss: 0.002955

Test set: Average loss: 0.0309, Accuracy: 9901/10000 (99%) 

Train Epoch: 12 [14848/60000 (25%)]	Loss: 0.012029
Train Epoch: 12 [30208/60000 (50%)]	Loss: 0.008490
Train Epoch: 12 [45568/60000 (75%)]	Loss: 0.002238

Test set: Average loss: 0.0285, Accuracy: 9913/10000 (99%) 

Train Epoch: 13 [14848/60000 (25%)]	Loss: 0.002908
Train Epoch: 13 [30208/60000 (50%)]	Loss: 0.001064
Train Epoch: 13 [45568/60000 (75%)]	Loss: 0.008539

Test set: Average loss: 0.0298, Accuracy: 9907/10000 (99%) 

Train Epoch: 14 [14848/60000 (25%)]	Loss: 0.000874
Train Epoch: 14 [30208/60000 (50%)]	Loss: 0.002521
Train Epoch: 14 [45568/60000 (75%)]	Loss: 0.004461

Test set: Average loss: 0.0299, Accuracy: 9905/10000 (99%) 

Train Epoch: 15 [14848/60000 (25%)]	Loss: 0.003159
Train Epoch: 15 [30208/60000 (50%)]	Loss: 0.001480
Train Epoch: 15 [45568/60000 (75%)]	Loss: 0.001043

Test set: Average loss: 0.0361, Accuracy: 9899/10000 (99%) 

Train Epoch: 16 [14848/60000 (25%)]	Loss: 0.005830
Train Epoch: 16 [30208/60000 (50%)]	Loss: 0.000970
Train Epoch: 16 [45568/60000 (75%)]	Loss: 0.002269

Test set: Average loss: 0.0297, Accuracy: 9913/10000 (99%) 

Train Epoch: 17 [14848/60000 (25%)]	Loss: 0.002263
Train Epoch: 17 [30208/60000 (50%)]	Loss: 0.010182
Train Epoch: 17 [45568/60000 (75%)]	Loss: 0.002237

Test set: Average loss: 0.0362, Accuracy: 9891/10000 (99%) 

Train Epoch: 18 [14848/60000 (25%)]	Loss: 0.009599
Train Epoch: 18 [30208/60000 (50%)]	Loss: 0.004443
Train Epoch: 18 [45568/60000 (75%)]	Loss: 0.013983

Test set: Average loss: 0.0323, Accuracy: 9917/10000 (99%) 

Train Epoch: 19 [14848/60000 (25%)]	Loss: 0.001589
Train Epoch: 19 [30208/60000 (50%)]	Loss: 0.001786
Train Epoch: 19 [45568/60000 (75%)]	Loss: 0.001769

Test set: Average loss: 0.0343, Accuracy: 9913/10000 (99%) 

Train Epoch: 20 [14848/60000 (25%)]	Loss: 0.001789
Train Epoch: 20 [30208/60000 (50%)]	Loss: 0.001748
Train Epoch: 20 [45568/60000 (75%)]	Loss: 0.000423

Test set: Average loss: 0.0344, Accuracy: 9903/10000 (99%) '''

train_losses = []
test_losses = []

for i in range(len(string)):
    if string[i:i + 4] == 'Loss':
        train_losses.append(float(string[i + 5: i + 14]))

print(train_losses)
for i in range(len(string)):
    if string[i:i + 4] == 'loss':
        test_losses.append(float(string[i + 6: i + 12]))

print(test_losses)

train_counter = [i * 12000 for i in range(1, len(train_losses) + 1)]
test_counter = [i * 36000 for i in range(0, len(test_losses))]
print(train_counter)
print(test_counter)


def draw_loss():
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red', s=15)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('times of training')
    plt.ylabel('negative log likelihood loss')
    plt.show()


draw_loss()
