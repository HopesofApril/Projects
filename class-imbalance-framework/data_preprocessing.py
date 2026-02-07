"""
데이터 전처리 코드 : 데이터 불러오기, 가공 및 정제, VIF 기반 변수 선정, 데이터 표준화 과정이 포함되어 있습니다.
코드 실행 시 X_train, X_test, y_train, y_test가 반환됩니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 데이터 불러오기
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# 2. 가공 및 정제
def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan) # 행에 포함된 inf 값을 NaN으로 변환
    df = df.dropna().reset_index(drop=True) # NaN 값이 포함된 행 제거
    return df
  
def encode_labels(
    df: pd.DataFrame,
    label_column: str,
    mapping: dict = None
) -> pd.DataFrame:
    if mapping: # mapping이 주어지면 해당 매핑을 활용해 변환 (ex. "featureA" : 1)
        df[label_column] = df[label_column].map(mapping)
    else: # 아니면 자동으로 factorize
        df[label_column], _ = pd.factorize(df[label_column])
    return df

# 2-1. 범주형 변수 원-핫 인코딩
def encode_categorical_features(
    df: pd.DataFrame,
    label_column: str
) -> pd.DataFrame:
    feature_df = df.drop(columns=[label_column])
    target_df = df[[label_column]]
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
    df_encoded = pd.concat([feature_df, target_df], axis=1)
    return df_encoded

# 3. VIF 기반 변수 선정 (10보다 클 경우 독립변수 간 상관관계가 높다고 판단하여 제거)
def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif.sort_values("VIF", ascending=False)

def drop_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    X = X.copy()
    while True:
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif < threshold:
            break
        drop_feature = vif.iloc[0]["feature"]
        X = X.drop(columns=[drop_feature])
    return X
  
# 4. 데이터 표준화 (이때, 수치형 변수만 표준화)
def scale_data(X_train: pd.DataFrame,
               X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include="number").columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler

# 5. 전체 파이프라인
def run_preprocessing(
    data_path: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    vif_threshold: float = 10.0,
    label_mapping: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # 데이터 불러오기
    df = load_data(data_path)

    # 가공 및 정제
    df = remove_invalid_values(df)
    df = encode_labels(df, label_column=label_column, mapping=label_mapping)
    df = encode_categorical_features(df, label_column=label_column)

    # 독립-종속변수 분리
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # 훈련-테스트 셋 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    # VIF 기반 변수 제거
    X_train = drop_high_vif(X_train, threshold=vif_threshold)
    X_test = X_test[X_train.columns]

    # 데이터 표준화
    X_train, X_test, _ = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test
