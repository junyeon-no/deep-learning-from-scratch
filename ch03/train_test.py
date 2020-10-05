import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist


(x_train,t_train), (x_test, t_test_ ) = load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)