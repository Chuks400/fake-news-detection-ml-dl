"""
Fake News Classification - Multi-Algorithm Comparison
======================================================
Dataset: fake_news.arff (44,899 news articles, binary: fake/real)
Algorithms compared:
  1. Naive Bayes (MultinomialNB)
  2. Logistic Regression
  3. Support Vector Machine (LinearSVC)
  4. Random Forest
  5. Decision Tree (J48 equivalent)
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("FAKE NEWS CLASSIFICATION - MULTI-ALGORITHM COMPARISON")
print("=" * 70)
print("\n[1] Loading dataset from fake_news.arff ...")

# Parse ARFF manually since scipy struggles with string attributes
texts = []
labels = []
with open("fake_news.arff", "r", encoding="utf-8") as f:
    data_started = False
    for line in f:
        if line.strip().upper() == "@DATA":
            data_started = True
            continue
        if data_started and line.strip():
            # Each line: 'text','date',class
            # Find the last comma before fake/real
            stripped = line.strip()
            if stripped.endswith("fake"):
                labels.append("fake")
                # Remove the trailing ,fake
                content = stripped[:-5]
            elif stripped.endswith("real"):
                labels.append("real")
                # Remove the trailing ,real
                content = stripped[:-5]
            else:
                continue
            # Extract text (first quoted field)
            if content.startswith("'"):
                # Find end of first quoted string
                end_idx = content.find("','")
                if end_idx != -1:
                    text = content[1:end_idx]
                else:
                    text = content[1:]
            else:
                text = content
            texts.append(text)

print(f"    Total instances loaded: {len(texts)}")
print(f"    Class distribution:")
fake_count = labels.count("fake")
real_count = labels.count("real")
print(f"      - Fake: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
print(f"      - Real: {real_count} ({real_count/len(labels)*100:.1f}%)")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n[2] Preprocessing text data ...")
print("    - Using TF-IDF Vectorization (max 50,000 features, bigrams)")

# Keep as lists (avoid massive numpy string array allocation)
y = np.array(labels)

# Train/Test split (70/30 stratified)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=0.30, random_state=42, stratify=y
)

print(f"    Training set: {len(X_train_text)} instances")
print(f"    Test set:     {len(X_test_text)} instances")

# TF-IDF Vectorization
print("    Fitting TF-IDF vectorizer ...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print(f"    Feature matrix shape: {X_train.shape}")

# =============================================================================
# 3. DEFINE CLASSIFIERS
# =============================================================================
print("\n[3] Configuring classifiers ...")

classifiers = {
    "1. Naive Bayes (MultinomialNB)": MultinomialNB(alpha=1.0),
    "2. Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    "3. Support Vector Machine (LinearSVC)": LinearSVC(
        max_iter=2000, C=1.0, random_state=42
    ),
    "4. Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "5. Decision Tree (J48)": DecisionTreeClassifier(
        criterion='entropy', random_state=42
    ),
}

for name in classifiers:
    print(f"    - {name}")

# =============================================================================
# 4. TRAIN, EVALUATE, AND COMPARE
# =============================================================================
print("\n[4] Training and evaluating each classifier ...")
print("=" * 70)

results = []
confusion_matrices = {}

for name, clf in classifiers.items():
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    
    # Train
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='fake', average='binary')
    rec = recall_score(y_test, y_pred, pos_label='fake', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='fake', average='binary')
    
    # Weighted averages (for multi-class fairness)
    prec_w = precision_score(y_test, y_pred, average='weighted')
    rec_w = recall_score(y_test, y_pred, average='weighted')
    f1_w = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])
    confusion_matrices[name] = cm
    
    print(f"\n  Accuracy:           {acc*100:.2f}%")
    print(f"  Precision (fake):   {prec*100:.2f}%")
    print(f"  Recall (fake):      {rec*100:.2f}%")
    print(f"  F1-Score (fake):    {f1*100:.2f}%")
    print(f"  Precision (weighted): {prec_w*100:.2f}%")
    print(f"  Recall (weighted):    {rec_w*100:.2f}%")
    print(f"  F1-Score (weighted):  {f1_w*100:.2f}%")
    print(f"  Training time:      {train_time:.2f}s")
    print(f"  Prediction time:    {predict_time:.2f}s")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Fake    Real")
    print(f"  Actual Fake  [ {cm[0][0]:>5}   {cm[0][1]:>5} ]")
    print(f"  Actual Real  [ {cm[1][0]:>5}   {cm[1][1]:>5} ]")
    
    print(f"\n  Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['fake', 'real'], digits=4))
    
    results.append({
        'Algorithm': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)': round(prec_w * 100, 2),
        'Recall (%)': round(rec_w * 100, 2),
        'F1-Score (%)': round(f1_w * 100, 2),
        'Train Time (s)': round(train_time, 2),
        'Predict Time (s)': round(predict_time, 2),
    })

# =============================================================================
# 5. CROSS-VALIDATION (5-Fold) FOR ROBUSTNESS
# =============================================================================
print("\n" + "=" * 70)
print("[5] 5-Fold Stratified Cross-Validation (on full dataset)")
print("=" * 70)

# Re-vectorize full dataset for cross-validation
print("    Vectorizing full dataset ...")
X_full = tfidf.fit_transform(texts)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, clf in classifiers.items():
    print(f"\n    Running CV for: {name} ...", end=" ")
    scores = cross_val_score(clf, X_full, y, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_cv = scores.mean() * 100
    std_cv = scores.std() * 100
    print(f"Accuracy: {mean_cv:.2f}% (+/- {std_cv:.2f}%)")
    cv_results.append({
        'Algorithm': name,
        'CV Accuracy Mean (%)': round(mean_cv, 2),
        'CV Accuracy Std (%)': round(std_cv, 2),
    })

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("[6] GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# 1. Fake vs Real News Distribution
print("\n    Generating Fake vs Real News Distribution chart...")
plt.figure(figsize=(8, 6))
class_counts = [fake_count, real_count]
class_labels = ['Fake', 'Real']
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(class_labels, class_counts, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Fake vs Real News Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Articles', fontsize=12)
plt.xlabel('News Class', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,}\n({height/len(labels)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fake_vs_real_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fake_vs_real_distribution.png")

# 2. Confusion Matrices for each classifier
print("\n    Generating Confusion Matrices...")
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'{name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    # Clean filename
    clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
    plt.savefig(f'confusion_matrix_{clean_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: confusion_matrix_{clean_name}.png")

# 3. Training Time Comparison
print("\n    Generating Training Time Comparison chart...")
plt.figure(figsize=(12, 6))
train_times = [r['Train Time (s)'] for r in results]
algo_names = [r['Algorithm'].replace('\n', ' ') for r in results]
colors = plt.cm.Set3(range(len(algo_names)))
bars = plt.bar(algo_names, train_times, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Training Time Comparison of Machine Learning Algorithms', fontsize=14, fontweight='bold')
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.xlabel('Algorithm', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: training_time_comparison.png")

# 4. F1-Score Comparison
print("\n    Generating F1-Score Comparison chart...")
plt.figure(figsize=(12, 6))
f1_scores = [r['F1-Score (%)'] for r in results]
colors = plt.cm.Set3(range(len(algo_names)))
bars = plt.bar(algo_names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
plt.title('F1-Score Comparison of Machine Learning Algorithms', fontsize=14, fontweight='bold')
plt.ylabel('F1-Score (%)', fontsize=12)
plt.xlabel('Algorithm', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(80, 100)  # Adjust based on your actual scores
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: f1_score_comparison.png")

# 5. Accuracy Comparison
print("\n    Generating Accuracy Comparison chart...")
plt.figure(figsize=(12, 6))
accuracies = [r['Accuracy (%)'] for r in results]
colors = plt.cm.Set3(range(len(algo_names)))
bars = plt.bar(algo_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Accuracy Comparison of Machine Learning Algorithms', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Algorithm', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(80, 100)  # Adjust based on your actual scores
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: accuracy_comparison.png")

print("\n    All visualizations generated successfully!")

# =============================================================================
# 7. FINAL COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("[7] FINAL COMPARISON SUMMARY")
print("=" * 70)

# Merge results
df_results = pd.DataFrame(results)
df_cv = pd.DataFrame(cv_results)
df_final = pd.merge(df_results, df_cv, on='Algorithm')

# Sort by F1-Score
df_final = df_final.sort_values('F1-Score (%)', ascending=False).reset_index(drop=True)

print("\n" + df_final.to_string(index=False))

# Determine the best algorithm
best = df_final.iloc[0]
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
Based on the comprehensive evaluation across multiple metrics:

  MOST EFFECTIVE ALGORITHM: {best['Algorithm']}
  
  - Test Accuracy:        {best['Accuracy (%)']}%
  - Weighted Precision:   {best['Precision (%)']}%
  - Weighted Recall:      {best['Recall (%)']}%
  - Weighted F1-Score:    {best['F1-Score (%)']}%
  - CV Accuracy (5-fold): {best['CV Accuracy Mean (%)']}% (+/- {best['CV Accuracy Std (%)']:.2f}%)
  - Training Time:        {best['Train Time (s)']}s
  - Prediction Time:      {best['Predict Time (s)']}s

Key Findings:
  - The best classifier achieves {best['F1-Score (%)']}% weighted F1-Score
  - Cross-validation confirms generalization (low std = stable performance)
  - Trade-off: Some models may train faster but sacrifice accuracy
  
Ranking (by Weighted F1-Score):
""")

for i, row in df_final.iterrows():
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {i+1}. {row['Algorithm']}: F1={row['F1-Score (%)']}%, Acc={row['Accuracy (%)']}%{marker}")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
