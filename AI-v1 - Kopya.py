import pandas as pd

# Veri setini yükle
columns = ['Letter', 'X-box', 'Y-box', 'Width', 'High', 'Onpix', 'X-bar', 'Y-bar',
           'X2bar', 'Y2bar', 'XYbar', 'X2ybr', 'XY2br', 'X-ege', 'Xegvy', 'Y-ege', 'Yegvx']
data = pd.read_csv('C:/Users/user/Documents/GitHub/A-Z---AI---Probability/letter_recognition.data', header=None, names=columns)

# Eğitim ve test setlerini ayır
train_data = data.iloc[:16000]  # İlk 16.000 satır
test_data = data.iloc[16000:]   # Son 4.000 satır

# Eğitim verisinin genel istatistiklerini kontrol et
print(train_data.describe())
# Öncül olasılıkları hesapla
prior_probs = train_data['Letter'].value_counts(normalize=True)
print(prior_probs)
def calculate_conditional_probs(train, feature, laplace_smoothing=True):
    conditional_probs = {}
    feature_values = train[feature].unique()
    letters = train['Letter'].unique()
    
    for letter in letters:
        letter_data = train[train['Letter'] == letter]
        letter_count = len(letter_data)
        probs = {}
        for value in feature_values:
            count = len(letter_data[letter_data[feature] == value])
            if laplace_smoothing:
                prob = (count + 1) / (letter_count + len(feature_values))
            else:
                prob = count / letter_count if letter_count > 0 else 0
            probs[value] = prob
        conditional_probs[letter] = probs
    return conditional_probs

# Koşullu olasılıkları hesapla
conditional_probs = {}
for feature in columns[1:]:
    conditional_probs[feature] = calculate_conditional_probs(train_data, feature)
import numpy as np

def predict(test, prior_probs, conditional_probs):
    predictions = []
    letters = list(prior_probs.index)
    for _, row in test.iterrows():
        letter_scores = {}
        for letter in letters:
            # Başlangıç log olasılık: log(P(Letter))
            score = np.log(prior_probs[letter])
            for feature in columns[1:]:
                value = row[feature]
                if value in conditional_probs[feature][letter]:
                    score += np.log(conditional_probs[feature][letter][value])
            letter_scores[letter] = score
        # Maksimum log olasılık değerine sahip harfi seç
        predicted_letter = max(letter_scores, key=letter_scores.get)
        predictions.append(predicted_letter)
    return predictions

# Tahminleri gerçekleştir
test_predictions = predict(test_data, prior_probs, conditional_probs)
from sklearn.metrics import confusion_matrix, accuracy_score

# Başarı oranı
test_accuracy = accuracy_score(test_data['Letter'], test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Karmaşa matrisi
conf_matrix = confusion_matrix(test_data['Letter'], test_predictions, labels=prior_probs.index)

# Çizdirme
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=False, fmt="d", xticklabels=prior_probs.index, yticklabels=prior_probs.index, cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Öncül olasılıkların histogramı
prior_probs.plot(kind='bar', figsize=(10, 5))
plt.title('Prior Probabilities of Letters')
plt.xlabel('Letter')
plt.ylabel('Probability')
plt.show()

# 5 harf ve 3 özellik için koşullu dağılımlar
selected_letters = ['A', 'B', 'C', 'D', 'E']
selected_features = ['X-box', 'Y-box', 'Width']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(selected_features):
    plt.subplot(3, 1, i + 1)
    for letter in selected_letters:
        letter_data = train_data[train_data['Letter'] == letter]
        sns.histplot(letter_data[feature], kde=True, label=f'Letter {letter}', stat='density', bins=16)
    plt.title(f'Distribution of {feature} for Selected Letters')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()
