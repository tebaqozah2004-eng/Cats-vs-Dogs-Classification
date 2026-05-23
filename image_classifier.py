import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ============================================
# 1. LOAD DATASET
# ============================================
cat_path = "dataset/cat"
dog_path = "dataset/dog"

images = []
labels = []

print("=" * 60)
print("CATS VS DOGS IMAGE CLASSIFICATION PROJECT")
print("=" * 60)

print("\n[1] Loading dataset...")

cat_count = 0
for file in os.listdir(cat_path):
    img = cv2.imread(os.path.join(cat_path, file))
    if img is not None:
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(0)
        cat_count += 1

dog_count = 0
for file in os.listdir(dog_path):
    img = cv2.imread(os.path.join(dog_path, file))
    if img is not None:
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(1)
        dog_count += 1

print(f"    Loaded {cat_count} cats, {dog_count} dogs")
print(f"    Total: {len(images)} images")

# ============================================
# 2. IMAGE ENHANCEMENT
# ============================================
print("\n[2] Image Enhancement (Brightness, Blur, Sharpening)...")


def adjust_brightness(image, alpha=1.3, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


sample_cat = images[0]
sample_dog = images[-1]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Image Enhancement - Before vs After', fontsize=16)

axes[0, 0].imshow(cv2.cvtColor(sample_cat, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Cat - Original')
axes[0, 0].axis('off')

cat_bright = adjust_brightness(sample_cat)
axes[0, 1].imshow(cv2.cvtColor(cat_bright, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Cat - Brightness')
axes[0, 1].axis('off')

cat_blur = apply_blur(sample_cat)
axes[0, 2].imshow(cv2.cvtColor(cat_blur, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Cat - Blur')
axes[0, 2].axis('off')

cat_sharp = apply_sharpening(sample_cat)
axes[0, 3].imshow(cv2.cvtColor(cat_sharp, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('Cat - Sharpening')
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(sample_dog, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Dog - Original')
axes[1, 0].axis('off')

dog_bright = adjust_brightness(sample_dog)
axes[1, 1].imshow(cv2.cvtColor(dog_bright, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Dog - Brightness')
axes[1, 1].axis('off')

dog_blur = apply_blur(sample_dog)
axes[1, 2].imshow(cv2.cvtColor(dog_blur, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Dog - Blur')
axes[1, 2].axis('off')

dog_sharp = apply_sharpening(sample_dog)
axes[1, 3].imshow(cv2.cvtColor(dog_sharp, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('Dog - Sharpening')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('image_enhancement_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 3. HISTOGRAM FEATURES
# ============================================
print("\n[3] Extracting Histogram Features...")


def extract_histogram_features(image, bins=16):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = []
    for channel in range(3):
        hist = cv2.calcHist([img_rgb], [channel], None, [bins], [0, 256])
        hist = hist / np.sum(hist)
        features.extend(hist.flatten())
    return np.array(features)


feature_list = []
for i, img in enumerate(images):
    feature_list.append(extract_histogram_features(img))

X = np.array(feature_list)
y = np.array(labels)

print(f"    Feature matrix: {X.shape} (48 features per image)")

# ============================================
# 4. TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n[4] Data split:")
print(f"    Training: {X_train.shape[0]} images")
print(f"    Testing: {X_test.shape[0]} images")

# ============================================
# 5. TRAIN MODELS
# ============================================
print("\n[5] Training models...")

print("    Training KNN...")
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("    Training SVM...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# ============================================
# 6. EVALUATION
# ============================================
def evaluate(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {name}")
    print(f"{'=' * 50}")
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Cat    Dog")
    print(f"Actual    Cat   {tn:4d}   {fp:4d}")
    print(f"          Dog   {fn:4d}   {tp:4d}")
    print("\nMetrics:")
    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}


results_knn = evaluate("KNN", y_test, y_pred_knn)
results_svm = evaluate("SVM", y_test, y_pred_svm)

# ============================================
# 7. COMPARISON TABLE
# ============================================
print(f"\n{'=' * 60}")
print("COMPARISON TABLE")
print(f"{'=' * 60}")
print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print(f"{'-' * 52}")
print(f"{'KNN':<12} {results_knn['accuracy']*100:<10.2f} "
      f"{results_knn['precision']*100:<10.2f} "
      f"{results_knn['recall']*100:<10.2f} "
      f"{results_knn['f1']*100:<10.2f}")
print(f"{'SVM':<12} {results_svm['accuracy']*100:<10.2f} "
      f"{results_svm['precision']*100:<10.2f} "
      f"{results_svm['recall']*100:<10.2f} "
      f"{results_svm['f1']*100:<10.2f}")

best = "KNN" if results_knn['accuracy'] > results_svm['accuracy'] else "SVM"
best_acc = max(results_knn['accuracy'], results_svm['accuracy']) * 100
print(f"\nBest Model: {best} ({best_acc:.2f}%)")

# ============================================
# 8. VISUALIZE COMPARISON
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.35

knn_vals = [results_knn['accuracy'], results_knn['precision'], results_knn['recall'], results_knn['f1']]
svm_vals = [results_svm['accuracy'], results_svm['precision'], results_svm['recall'], results_svm['f1']]

bars1 = ax.bar(x - width/2, knn_vals, width, label='KNN', color='#3498db')
bars2 = ax.bar(x + width/2, svm_vals, width, label='SVM', color='#e74c3c')

for bar, val in zip(bars1, knn_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=10)
for bar, val in zip(bars2, svm_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=10)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('KNN vs SVM Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 9. CONFUSION MATRICES
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, cm) in enumerate([('KNN', results_knn['cm']), ('SVM', results_svm['cm'])]):
    axes[idx].imshow(cm, cmap=plt.cm.Blues if idx == 0 else plt.cm.Greens)
    axes[idx].set_title(f'Confusion Matrix - {name}')
    axes[idx].set_xticks([0, 1])
    axes[idx].set_yticks([0, 1])
    axes[idx].set_xticklabels(['Cat', 'Dog'])
    axes[idx].set_yticklabels(['Cat', 'Dog'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            axes[idx].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 10. FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"""
Lectures Covered:
- Lecture 8: Image processing ✓
- Lecture 9: Image Enhancement ✓
- Lecture 10: Histogram Features ✓
- Lecture 16-19: KNN ✓
- Lecture 17: Metrics ✓
- Lecture 21: SVM ✓

Results:
KNN Accuracy: {results_knn['accuracy'] * 100:.2f}%
SVM Accuracy: {results_svm['accuracy'] * 100:.2f}%

Generated Files:
- image_enhancement_results.png
- models_comparison.png
- confusion_matrices.png

PROJECT READY FOR SUBMISSION
""")