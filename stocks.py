import matplotlib.pyplot as plt
import seaborn as sns

acc=[99.82591431678016,55.0063821613473,89.6569164]
model=['LSTM','Dense','LinearRegression']

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')
plt.show()
