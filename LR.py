from pstats import SortKey
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# veriyi yukle
letters = pd.read_csv('C:/Users/ceyda/codes/ai/hw1/letter_recognition.data')

# veriyi egitim ve test olarak ayir
train = letters.head(16000)
test = letters.tail(4000)

# bazi bilgileri kontrol amacli bastir
print(train.Letter)
print(train.values[0:10,:])
data = train.values
L = train.Letter.values
print(test.values[0:10,:])

# egitim verisini grupla ve oncul olasılıkları P(letter) hesapla
letter_num =  train.groupby('Letter')['Letter'].count()
print(letter_num)
letter_counts =  letter_num.values
letter_sum = sum(letter_counts)
prior_prob = letter_counts/letter_sum

# egitim verisini grupla ve kosullu olasilik modellerini olustur
features = train.columns
num_features = features.shape[0]

for i in range(1,num_features):
    fi =  train.groupby(['Letter', features[i]])[features[i]].count()
    print(fi.values)
    print(fi.index.values)

indices = train.Letter == 'A'
print(data[indices,:])

# test verisini olasilik modelleriyle degerlendir

# gercek etiketlerle tahmini etiketleri karsilastirip başarıyı ölç



