import os
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_score,\
        recall_score, f1_score
from lightgbm import LGBMClassifier


mlflow.set_tracking_uri("http://89.223.66.144:5000")
mlflow.set_experiment('mlflow_test_lgbm')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://89.223.66.144:9000'
os.environ['MLflow_TRACKING_USERNAME'] = 'MLflow-user'

ml_config_train = "train_config.json"
ml_config_test = "test_config.json"
data_config_train = ml_config_train.get("data_config", None)
data_config_test = ml_config_test.get("data_config", None)

mlflow.lighgbm.autolog()
with mlflow.start_run():

    df_train = data_config_train.get("train_data", None)
    df_test = data_config_test.get("test_data", None)
    target_column = data_config_train.get("target_column", "Rating")
    data_column = data_config_train.get("data_column", "Review")

    word_vect = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=2000)

    word_vect.fit(df_train[data_column])
    train_word_features = word_vect.transform(df_train[data_column])
    test_word_features = word_vect.transform(df_test[data_column])

    X_train = train_word_features.tocsr()
    X_test = test_word_features.tocsr()
    y_train = df_train[target_column]
    y_test = df_test[target_column]

    clf = LGBMClassifier(random_state=42)
    clf.fit(X_train, y_train)

    predictions_train = clf.predict(X_train)
    probas_train = clf.predict_proba(X_train)
    predictions_test = clf.predict(X_test)
    probas_test = clf.predict_proba(X_test)

    signature = infer_signature(X_train, predictions_train)

    model_info = mlflow.lightgbm.log_model(clf, "lgbm", signature=signature)

    metrics_train = {
                 "roc_auc": roc_auc_score(y_train, probas_train[:, 1]),
                 "precision_macro": precision_score(y_train, predictions_train,
                                                    average='macro'),
                 "recall_macro": recall_score(y_train, predictions_train,
                                              average='macro'),
                 "f1_macro": f1_score(y_train, predictions_train,
                                      average='macro')
                }

    metrics_test = {
                "roc_auc": roc_auc_score(y_test, probas_test[:, 1]),
                "precision_macro": precision_score(y_test, predictions_test,
                                                   average='macro'),
                "recall_macro": recall_score(y_test, predictions_test,
                                             average='macro'),
                "f1_macro": f1_score(y_test, predictions_test, average='macro')
                }

    mlflow.log_metrics(metrics_train)
    mlflow.log_metrics(metrics_test)

    autolog_run = mlflow.last_active_run()
