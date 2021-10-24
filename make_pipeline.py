import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Artifact,
    OutputPath,
    InputPath,
    ClassificationMetrics,
    Model,
)
from typing import List

@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/download_csv.yaml',
)
def download_csv(url: str, output_csv: Output[Dataset]):
    import urllib.request
    import pandas as pd
    
    urllib.request.urlretrieve(url=url,
                               filename=output_csv.path,
                              )


@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/train_test_split.yaml',
)
def train_test_split(input_csv: Input[Dataset],
                     seed: int,
                     target: str,
                     train_csv: Output[Dataset],
                     test_csv: Output[Dataset]
                    ):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = pd.read_csv(input_csv.path)
    train, test = train_test_split(df,
                                   test_size=0.2,
                                   shuffle=True,
                                   random_state=seed,
                                   stratify=df[target],
                                  )
    train_df = pd.DataFrame(train)
    train_df.columns = df.columns
    test_df = pd.DataFrame(test)
    test_df.columns = df.columns
    
    train_df.to_csv(train_csv.path, index=False)
    test_df.to_csv(test_csv.path, index=False)


@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/preprocessing.yaml',
)
def preprocessing(input_csv: Input[Dataset],
                  numerical_features: str,
                  categorical_features: str,
                  target: str,
                  features: Output[Dataset],
                  labels: Output[Dataset],
                  scaler_obj: Output[Model],
                  encoder_obj: Output[Model],
                 ):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from pickle import dump
    from ast import literal_eval
    
    # The features are stored as strings. We do the trick with literal_eval to convert them to the
    # correct format (list)
    categorical_features = literal_eval(categorical_features)
    numerical_features = literal_eval(numerical_features)
    
    df = pd.read_csv(input_csv.path)
    X_cat = df[categorical_features]
    X_num = df[numerical_features]
    y = df[target]
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    encoder = OneHotEncoder()
    X_cat = encoder.fit_transform(X_cat).toarray()
    X = np.concatenate([X_num, X_cat], axis=1)
    
    pd.DataFrame(X).to_csv(features.path, index=False)
    y.to_csv(labels.path, index=False)
    # To prevent leakage, the scaler and the encoder should not see the test dataset.
    # We save the scaler and encoder that have been fit to the training dataset and 
    # use it directly on the test dataset later on.
    dump(scaler, open(scaler_obj.path, 'wb'))
    dump(encoder, open(encoder_obj.path, 'wb'))


@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/logistic_regression.yaml',
)
def logistic_regression(features: Input[Dataset],
                        labels: Input[Dataset],
                        param_grid: str,
                        num_folds: int,
                        scoring: str,
                        seed: int,
                        best_model: Output[Model],
                        best_params: Output[Dataset],
                        best_score: Output[Metrics],
                        cv_results: Output[Dataset],
                       ) -> float:
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from pickle import dump
    from ast import literal_eval
    
    lr = LogisticRegression(solver='liblinear',
                            random_state=seed,
                           )
    param_grid = literal_eval(param_grid)
    grid_search = GridSearchCV(lr,
                               param_grid=param_grid,
                               scoring=scoring,
                               refit=True, # Use the whole dataset to retrain after finding the best params
                               cv=num_folds,
                               verbose=2,
                              )
    
    X, y = pd.read_csv(features.path).values, pd.read_csv(labels.path).values
    grid_search.fit(X, y)
    
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results.path, index=False)
    best_params_ = grid_search.best_params_
    for key, value in best_params_.items():
        best_params_[key] = [value]
    pd.DataFrame(best_params_).to_csv(best_params.path, index=False)
    dump(grid_search.best_estimator_, open(best_model.path, 'wb'))
    best_score.log_metric(scoring, grid_search.best_score_)
    
    return grid_search.best_score_
    
    
@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/random_forests.yaml',
)
def random_forests(features: Input[Dataset],
                   labels: Input[Dataset],
                   param_grid: str,
                   num_folds: int,
                   scoring: str,
                   seed: int,
                   best_model: Output[Model],
                   best_params: Output[Dataset],
                   best_score: Output[Metrics],
                   cv_results: Output[Dataset],
                  ) -> float:
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from pickle import dump
    from ast import literal_eval
    
    rf = RandomForestClassifier(random_state=seed)
    param_grid = literal_eval(param_grid)
    grid_search = GridSearchCV(rf,
                               param_grid=param_grid,
                               scoring=scoring,
                               refit=True, # Use the whole dataset to retrain after finding the best params
                               cv=num_folds,
                               verbose=2,
                              )
    
    X, y = pd.read_csv(features.path).values, pd.read_csv(labels.path).values
    grid_search.fit(X, y)
    
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results.path, index=False)
    best_params_ = grid_search.best_params_
    for key, value in best_params_.items():
        best_params_[key] = [value]
    pd.DataFrame(best_params_).to_csv(best_params.path, index=False)
    dump(grid_search.best_estimator_, open(best_model.path, 'wb'))
    best_score.log_metric(scoring, grid_search.best_score_)
    
    return grid_search.best_score_


@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/k_nearest_neighbors.yaml',
)
def knn(features: Input[Dataset],
        labels: Input[Dataset],
        param_grid: str,
        num_folds: int,
        scoring: str,
        best_model: Output[Model],
        best_params: Output[Dataset],
        best_score: Output[Metrics],
        cv_results: Output[Dataset],
       ) -> float:
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from pickle import dump
    from ast import literal_eval
    
    k_nn = KNeighborsClassifier()
    param_grid = literal_eval(param_grid)
    grid_search = GridSearchCV(k_nn,
                               param_grid=param_grid,
                               scoring=scoring,
                               refit=True, # Use the whole dataset to retrain after finding the best params
                               cv=num_folds,
                               verbose=2,
                              )
    
    X, y = pd.read_csv(features.path).values, pd.read_csv(labels.path).values
    grid_search.fit(X, y)
    
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results.path, index=False)
    best_params_ = grid_search.best_params_
    for key, value in best_params_.items():
        best_params_[key] = [value]
    pd.DataFrame(best_params_).to_csv(best_params.path, index=False)
    dump(grid_search.best_estimator_, open(best_model.path, 'wb'))
    best_score.log_metric(scoring, grid_search.best_score_)
    
    return grid_search.best_score_


@component(
    base_image='yinanli617/customer-churn:latest',
    output_component_file='./components/predict_test_data.yaml',
)
def predict_test_data(test_csv: Input[Dataset],
                      scaler_obj: Input[Model],
                      encoder_obj: Input[Model],
                      lr_model: Input[Model],
                      rf_model: Input[Model],
                      knn_model: Input[Model],
                      lr_score: float,
                      rf_score: float,
                      knn_score: float,
                      categorical_features: str,
                      numerical_features: str,
                      target: str,
                      metrics: Output[Metrics],
                     ):
    import pandas as pd
    import numpy as np
    from pickle import load
    from ast import literal_eval
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    
    categorical_features = literal_eval(categorical_features)
    numerical_features = literal_eval(numerical_features)
    
    df = pd.read_csv(test_csv.path)
    X_cat = df[categorical_features]
    X_num = df[numerical_features]
    y = df[target]
    
    scaler = load(open(scaler_obj.path, 'rb'))
    X_num = scaler.transform(X_num)
    encoder = load(open(encoder_obj.path, 'rb'))
    X_cat = encoder.transform(X_cat).toarray()
    X = np.concatenate([X_num, X_cat], axis=1)
    
    models_dict = {lr_score: lr_model,
                   rf_score: rf_model,
                   knn_score: knn_model,
                  }
    best_model = models_dict[max(models_dict.keys())]
    model = load(open(best_model.path, 'rb'))
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    
    metrics.log_metric('Accuracy', accuracy)
    metrics.log_metric('F1 score', f1)
    metrics.log_metric('Recall', recall)
    metrics.log_metric('Precision', precision)
    metrics.log_metric('ROC_AUC', roc_auc)


@dsl.pipeline(
    name='bank-customer-churn-pipeline',
    # You can optionally specify your own pipeline_root
    pipeline_root='gs://kfp-yli/customer-churn',
)
def my_pipeline(url: str,
                num_folds: int,
                target: str,
                numerical_features: str,
                categorical_features: str,
                scoring: str,
                logistic_regression_params: str,
                random_forests_params: str,
                knn_params: str,
                seed: int,
               ):

    download_csv_task = download_csv(url=url)

    train_test_split_task = train_test_split(input_csv=download_csv_task.outputs['output_csv'],
                                             seed=seed,
                                             target=target,
                                            )
    
    train_preprocessing_task = preprocessing(input_csv=train_test_split_task.outputs['train_csv'],
                                             numerical_features=numerical_features,
                                             categorical_features=categorical_features,
                                             target=target,
                                            )
    
    logistic_regression_task = logistic_regression(features=train_preprocessing_task.outputs['features'],
                                                   labels=train_preprocessing_task.outputs['labels'],
                                                   scoring=scoring,
                                                   seed=seed,
                                                   num_folds=num_folds,
                                                   param_grid=logistic_regression_params,
                                                  )

    random_forests_task = random_forests(features=train_preprocessing_task.outputs['features'],
                                         labels=train_preprocessing_task.outputs['labels'],
                                         scoring=scoring,
                                         seed=seed,
                                         num_folds=num_folds,
                                         param_grid=random_forests_params,
                                        )
    
    knn_task = knn(features=train_preprocessing_task.outputs['features'],
                   labels=train_preprocessing_task.outputs['labels'],
                   scoring=scoring,
                   num_folds=num_folds,
                   param_grid=knn_params,
                  )
    
    predict_test_data_task = predict_test_data(test_csv=train_test_split_task.outputs['test_csv'],
                                               scaler_obj=train_preprocessing_task.outputs['scaler_obj'],
                                               encoder_obj=train_preprocessing_task.outputs['encoder_obj'],
                                               lr_model=logistic_regression_task.outputs['best_model'],
                                               rf_model=random_forests_task.outputs['best_model'],
                                               knn_model=knn_task.outputs['best_model'],
                                               lr_score=logistic_regression_task.outputs['output'],
                                               rf_score=random_forests_task.outputs['output'],
                                               knn_score=knn_task.outputs['output'],
                                               categorical_features=categorical_features,
                                               numerical_features=numerical_features,
                                               target=target
                                              )

if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=my_pipeline,
        package_path='./pipeline/customer-churn_pipeline.yaml')