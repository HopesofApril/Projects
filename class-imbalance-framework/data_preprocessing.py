'''
✏️ Example
------
from preprocessing import preprocessing_pipeline

X_train, X_test, y_train, y_test = preprocessing_pipeline(
    path="file-name.csv",
    label_col="y-label-name",
    vif_threshold=10
)
------
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 불러오기
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # 컬럼 공백 제거 (앞뒤 공백 문제 해결)
    return df

# 범주형 변수 수치화
def encode_features_and_label(df, label_col):
    df = df.copy()
    
    # 값이 2개라면(ex. 성별 등) 0, 1로 변환
    unique_vals = df[label_col].unique()
    if len(unique_vals) == 2:
        df[label_col] = df[label_col].astype("category").cat.codes
    else:
        df[label_col] = np.where(df[label_col] == unique_vals[0], 0, 1)
      
    # 값이 여러개라면 원-핫 인코딩
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in categorical_cols:
        categorical_cols.remove(label_col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# inf 값을 NaN으로 변환 후 NaN값 행 제거
def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df

# 수치형 변수 추출
def get_numeric_columns(X):
    return [
        col for col in X.select_dtypes(include=np.number).columns
        if X[col].nunique() > 2  # 범주형 수치화 컬럼 제외
    ]

# VIF 계산
def calculate_vif(X):
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_df.sort_values("VIF", ascending=False)

# VIF 값이 10 이상이라면 제거
def remove_high_vif(X, threshold=10.0):
    while True:
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            drop_feature = vif.iloc[0]["feature"]
            print(f"Removing '{drop_feature}' (VIF={max_vif:.2f})")
            X = X.drop(columns=[drop_feature])
        else:
            break
    return X

# 전체 파이프라인
def preprocessing_pipeline(
    path,
    label_col,
    test_size=0.2,
    vif_threshold=10.0,
    random_state=42
):

    df = load_data(path)  # 데이터 불러오기
    df = clean_data(df)  # 데이터 정제
    df = encode_features_and_label(df, label_col)  # 범주형 변수 수치화

    # 독립-종속 변수 분리
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # 수치형 컬럼만 선택
    num_cols = get_numeric_columns(X)
    
    # 훈련-테스트 셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 표준화
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # VIF 기반 변수 제거
    X_train = remove_high_vif(X_train, threshold=vif_threshold)
    X_test = X_test[X_train.columns]  # 테스트 셋도 동일하게
      
    return X_train, X_test, y_train, y_test
