import matplotlib.pyplot as plt

execution_times = [4.645, 1.838]
matching_accuracy = [64, 36]
labels = ['Convex Hull + CPD', 'Keypoints (ORB) + CPD']

fs=16

plt.subplot(2, 1, 1)
plt.bar(labels, execution_times, color='blue')
plt.ylabel('Execution Time (seconds)', fontsize=fs)
plt.title('Execution Time Comparison', fontsize=fs)
for i in range(len(labels)):
    plt.text(i, execution_times[i], str(execution_times[i]), ha='center', va='top', color='white', fontsize=fs)

plt.subplot(2, 1, 2)
plt.bar(labels, matching_accuracy, color='red')
plt.ylabel('Matching Accuracy (%)', fontsize=fs)
plt.title('Matching Accuracy Comparison', fontsize=fs)

for i in range(len(labels)):
    plt.text(i, matching_accuracy[i], str(matching_accuracy[i]), ha='center', va='top', color='white', fontsize=fs)
plt.tight_layout(pad=1.0)

plt.show()