import os
import yaml
import pandas as pd
import numpy as np
from datasets import load_dataset
from time import time
# 🔍 설정 경로
with open("params/data.yaml", "r") as f:
    config = yaml.safe_load(f)

repo_name = config["repo"]["name"]
data_files = config["repo"]["data_files"]
splits = config["repo"]["split"]

# 📦 데이터셋 불러오기 (파일 이름을 명시해서 불러와야 오류 없음)
print(time(), "데이터셋 불러오기 시작")
train_dataset = load_dataset(
    repo_name,
    data_files={
        "train": data_files["train"],
    },
    split=splits["train"]
)
print(time(), "train 데이터셋 불러오기 완료")

test_dataset = load_dataset(
    repo_name,
    data_files={
        "test": data_files["test"],
    },
    split=splits["test"]
)
print(time(), "test 데이터셋 불러오기 완료")
# 🔧 전처리 함수
def preprocess_split(split_name, df):
    label_col = df.columns[0]
    labels = df[label_col].astype(int)
    features = df.drop(columns=[label_col]).astype(np.float32) / 255.0
    return labels, features

# ✅ train 전처리
print(time(), "train 전처리 시작")
train_df = pd.DataFrame(train_dataset)
print(time(), "train 데이터프레임 생성 완료")
y_train, x_train = preprocess_split("train", train_df)
print(time(), "train 전처리 완료")

# ✅ test 전처리
print(time(), "test 전처리 시작")
test_df = pd.DataFrame(test_dataset)
print(time(), "test 데이터프레임 생성 완료")
y_test, x_test = preprocess_split("test", test_df)
print(time(), "test 전처리 완료")

# 📂 저장 경로
os.makedirs("data", exist_ok=True)
print(time(), "저장 경로 생성 시작")
np.save("data/x_train.npy", x_train.values)
np.save("data/y_train.npy", y_train.values)
np.save("data/x_test.npy", x_test.values)
np.save("data/y_test.npy", y_test.values)
print(time(), "저장 완료")

print("✅ 전처리 완료 및 저장 완료 (data/processed)")
