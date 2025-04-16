import os
import yaml
import mlflow
import numpy as np
import tensorflow as tf
from time import time
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MNIST_TEST")

# param + dataset config
with open("params/params.yaml") as f:
    config = yaml.safe_load(f)

model_params = config["model"]
train_params = config["train"]
data_params = config["dataset"]

run_name = f"{train_params['optimizer']}_lr{train_params['learning_rate']}"
print(time(), "파라미터 로드 완료")
# 데이터 불러오기 (이미 정제된 npy 파일!)
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")
print(time(), "데이터 로드 완료")
# 옵티마이저 설정
optimizer_name = train_params["optimizer"].lower()
if optimizer_name == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_params["learning_rate"])
elif optimizer_name == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=train_params["learning_rate"])
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(model_params["hidden_units"], activation='relu'),
    tf.keras.layers.Dropout(model_params["dropout"]),
    tf.keras.layers.Dense(data_params["num_classes"], activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# MLflow Run 시작
print("MLflow Run 시작")
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({**model_params, **train_params, **data_params})

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        verbose=2
    )

    for metric in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        if metric in history.history:
            for epoch, value in enumerate(history.history[metric]):
                mlflow.log_metric(metric, value, step=epoch)

    mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name=run_name)
