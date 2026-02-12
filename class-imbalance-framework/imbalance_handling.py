'''
✏️ Example
------
from imbalance_handling import imbalance_pipeline

balanced_datasets, X_test_balanced, y_test_balanced = imbalance_pipeline(
    X_train, X_test, y_train, y_test,
    n_datasets=4 # balanced dataset 개수 지정
)
------
'''

import pandas as pd
from sklearn.utils import resample

# 테스트 셋 균형 맞추기
def balance_test_set(X_test, y_test, random_state=42):
  test_df = pd.concat([X_test, y_test], axis=1)

  value_counts = test_df[y_test.name].value_counts()
  major_label = value_counts.idxmax()
  minor_label = value_counts.idxmin()

  df_major = test_df[test_df[y_test.name] == major_label]
  df_minor = test_df[test_df[y_test.name] == minor_label]
  
  # 소수 클래스 기준 언더샘플링
  df_major_down = df_major.sample(n=len(df_minor), random_state=random_state)
  test_balanced = pd.concat([df_major_down, df_minor]).sample(frac=1, random_state=random_state).reset_index(drop=True)
  X_test_balanced = test_balanced.drop(columns=[y_test.name])
  y_test_balanced = test_balanced[y_test.name]

  return X_test_balanced, y_test_balanced

# 균형 맞춘 다수의 훈련 셋 만들기
def create_balanced_train_sets(X_train, y_train, n_datasets=4, random_state=0):
  train_df = pd.concat([X_train, y_train], axis=1)

  value_counts = train_df[y_train.name].value_counts()
  major_label = value_counts.idxmax()
  minor_label = value_counts.idxmin()

  df_major = train_df[train_df[y_train.name] == major_label]
  df_minor = train_df[train_df[y_train.name] == minor_label]
  balanced_datasets = []

  for i in range(n_datasets):
    minor_sampled = resample(df_minor, replace=False, n_samples=len(df_minor), random_state=random_state + i)
    major_sampled = resample(df_major, replace=True, n_samples=len(minor_sampled), random_state=random_state + i)
    
    balanced_df = pd.concat([major_sampled, minor_sampled]).sample(frac=1, random_state=random_state + i).reset_index(drop=True)
    X_bal = balanced_df.drop(columns=[y_train.name])
    y_bal = balanced_df[y_train.name]
    balanced_datasets.append((X_bal, y_bal))

  return balanced_datasets

# 전체 파이프라인
def imbalance_pipeline(X_train, X_test, y_train, y_test, n_datasets=4, random_state=42):
  X_test_bal, y_test_bal = balance_test_set(X_test, y_test, random_state=random_state)
  balanced_datasets = create_balanced_train_sets(X_train, y_train, n_datasets=n_datasets, random_state=random_state)
  return balanced_datasets, X_test_bal, y_test_bal
