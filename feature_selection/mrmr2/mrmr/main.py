import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import jinja2
import numpy as np
import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

FLOOR = .001


def groupstats2fstat(avg, var, n):
    """Compute F-statistic of some variables across groups

    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.

    Parameters
    ----------
    avg: pandas.DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.

    var: pandas.DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.

    n: pandas.DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each variable, based on group statistics.

    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f.fillna(0.0)


def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False,
              return_scores=False, show_progress=True):
    """General function for mRMR algorithm.

    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K

    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.

    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.

    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.

    redundancy_args: dict (optional, default={])
        Optional arguments for redundancy_func.

    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()

    for i in tqdm(range(K), disable=not show_progress):

        score_numerator = relevance.loc[not_selected_features]

        if i > 0:

            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))

        else:
            score_denominator = pd.Series(1, index=features)

        score = score_numerator / score_denominator

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    if not return_scores:
        return selected_features
    else:
        return (selected_features, relevance, redundancy)


def mrmr_base2(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False,
              return_scores=False, show_progress=True):
    """General function for mRMR algorithm.

    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K

    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.

    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.

    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.

    redundancy_args: dict (optional, default={])
        Optional arguments for redundancy_func.

    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()

    for i in tqdm(range(K), disable=not show_progress):

        score_numerator = relevance.loc[not_selected_features]

        if i > 0:

            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))

        else:
            score_denominator = pd.Series(1, index=features)

        score = score_numerator - score_denominator

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    if not return_scores:
        return selected_features
    else:
        return (selected_features, relevance, redundancy)


def get_numeric_features(bq_client, table_id, target_column):
    """Get all numeric feature names from a BigQuery table

    Parameters
    ----------
    bq_client: google.cloud.bigquery.Client
        Google API's Client, already initialized with OAuth2 Credentials.

    table_id: str
        Unique ID of a Bigquery table, formatted as 'project_name.dataset_name.table_name'.
        Example: 'bigquery-public-data.baseball.games_wide'.

    target_column: str
        Name of target column.

    Returns
    -------
    numeric_features : list of str
        List of numeric feature names.
    """
    schema = bq_client.get_table(table_id).schema
    numeric_features = [field.name for field in schema if
                        field.field_type in ['INTEGER', 'FLOAT'] and field.name != target_column]
    return numeric_features


def correlation(target_column, features, bq_client, table_id):
    """Compute (Pearson) correlation between one numeric target column and many numeric columns of a BigQuery table

    Parameters
    ----------
    bq_client: google.cloud.bigquery.Client
        Google API's Client, already initialized with OAuth2 Credentials.

    table_id: str
        Unique ID of a Bigquery table, formatted as 'project_name.dataset_name.table_name'.
        Example: 'bigquery-public-data.baseball.games_wide'

    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    Returns
    -------
    corr: pandas.Series of shape (n_variables, )
        Correlation between each column and the target column.
    """
    jinja_query = """
{% set COLUMNS = features%}
SELECT
  {% for COLUMN in COLUMNS -%}
  CORR(target_column, {{COLUMN}}) AS {{COLUMN}}{% if not loop.last %},{% endif %}
  {% endfor -%}
FROM 
  table_id
    """ \
        .replace('table_id', table_id) \
        .replace('target_column', target_column) \
        .replace('features', str(features))

    corr = bq_client.query(query=jinja2.Template(jinja_query).render()).to_dataframe().iloc[0, :]
    corr.name = target_column

    return corr


def f_classif(target_column, features, bq_client, table_id):
    """Compute F-statistic of one (discrete) target column and many (discrete or continuous) numeric columns of a BigQuery table

    Parameters
    ----------
    bq_client: google.cloud.bigquery.Client
        Google API's Client, already initialized with OAuth2 Credentials.

    table_id: str
        Unique ID of a Bigquery table, formatted as 'project_name.dataset_name.table_name'.
        Example: 'bigquery-public-data.baseball.games_wide'

    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each numeric column grouped by the target column.
    """
    jinja_query = """
{% set COLUMNS = features%}
SELECT 
  target_column,
  {% for COLUMN in COLUMNS -%}
  METRIC(CAST({{COLUMN}} AS FLOAT64)) AS {{COLUMN}}{% if not loop.last %},{% endif %}
  {% endfor -%}
FROM 
  table_id
GROUP BY 
  target_column
    """ \
        .replace('table_id', table_id) \
        .replace('target_column', target_column) \
        .replace('features', str(features))

    avg = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'AVG')).render()
    ).to_dataframe().set_index(target_column, drop=True).astype(float)

    var = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'VAR_POP')).render()
    ).to_dataframe().set_index(target_column, drop=True).astype(float)

    n = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'COUNT')).render()
    ).to_dataframe().set_index(target_column, drop=True).astype(float)

    f = groupstats2fstat(avg=avg, var=var, n=n)
    f.name = target_column

    return f


def f_regression(target_column, features, bq_client, table_id):
    """Compute F-statistic between one numeric target column and many numeric columns of a BigQuery table

    F-statistic is actually obtained from the Pearson's correlation coefficient through the following formula:
    corr_coef ** 2 / (1 - corr_coef ** 2) * degrees_of_freedom
    where degrees_of_freedom = n_instances - 1.

    Parameters
    ----------
    bq_client: google.cloud.bigquery.Client
        Google API's Client, already initialized with OAuth2 Credentials.

    table_id: str
        Unique ID of a Bigquery table, formatted as 'project_name.dataset_name.table_name'.
        Example: 'bigquery-public-data.baseball.games_wide'

    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic between each column and the target column.
    """
    jinja_query = """
{% set COLUMNS = features%}
SELECT
  {% for COLUMN in COLUMNS -%}
  COUNTIF(target_column IS NOT NULL AND {{COLUMN}} IS NOT NULL) AS {{COLUMN}}{% if not loop.last %},{% endif %}
  {% endfor -%}
FROM 
  table_id
    """ \
        .replace('table_id', table_id) \
        .replace('target_column', target_column) \
        .replace('features', str(features))

    corr_coef = correlation(target_column=target_column, features=features, bq_client=bq_client, table_id=table_id)

    n = bq_client.query(query=jinja2.Template(jinja_query).render()).to_dataframe().iloc[0, :]
    n.name = target_column

    deg_of_freedom = n - 2
    corr_coef_squared = corr_coef ** 2
    f = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    return f



def mrmr_regression(bq_client, table_id, target_column, K,
                    features=None, denominator='mean', only_same_domain=False,
                    return_scores=False, show_progress=True):
    """MRMR feature selection for a regression task

    Parameters
    ----------
    bq_client: google.cloud.bigquery.Client
        Google API's Client, already initialized with OAuth2 Credentials.

    table_id: str
        Unique ID of a Bigquery table, formatted as 'project_name.dataset_name.table_name'.
        Example: 'bigquery-public-data.baseball.games_wide'

    target_column: str
        Name of target column.

    K: int
        Number of features to select.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected: list of str
        List of selected features.
    """
    if features is None:
        features = get_numeric_features(bq_client=bq_client, table_id=table_id, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'bq_client': bq_client, 'table_id': table_id, 'target_column': target_column,
                      'features': features}
    redundancy_args = {'bq_client': bq_client, 'table_id': table_id}

    return mrmr_base(K=K, relevance_func=f_regression, redundancy_func=correlation,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)

def parallel_df(func, df, series, n_jobs):
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def _f_classif(X, y):
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y, n_jobs):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs)


def f_regression(X, y, n_jobs):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs)


def _ks_classif(X, y):
    def _ks_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        x = x[x_not_na]
        y = y[x_not_na]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)


def ks_classif(X, y, n_jobs):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)

    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)


def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X


def mrmr_classif(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a classification task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "ks" (kolmogorov-smirnov), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    if relevance == "f":
        relevance_func = functools.partial(f_classif, n_jobs=n_jobs)
    elif relevance == "ks":
        relevance_func = functools.partial(ks_classif, n_jobs=n_jobs)
    elif relevance == "rf":
        relevance_func = random_forest_classif
    else:
        relevance_func = relevance

    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    fcq = mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)
    fcd = mrmr_base2(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)
    return fcq,fcd

def mrmr_regression(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a regression task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = functools.partial(f_regression, n_jobs=n_jobs) if relevance == 'f' else (
        random_forest_regression if relevance == 'rf' else relevance)
    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)


def get_numeric_features(df, target_column):
    """Get all numeric column names from a Spark DataFrame

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    target_column: str
        Name of target column.

    Returns
    -------
    numeric_features : list of str
        List of numeric column names.
    """
    numeric_dtypes = ['int', 'bigint', 'long', 'float', 'double', 'decimal']
    numeric_features = [column_name for column_name, column_type in df.dtypes if column_type in numeric_dtypes and column_name != target_column]
    return numeric_features


def correlation(target_column, features, df):
    out = pd.Series(features, index=features).apply(
        lambda feature: df.select([feature, target_column]).na.drop("any").corr(feature, target_column)
    ).astype(float).fillna(0.0)
    return out


def notna(target_column, features, df):
    out = pd.Series(features, index=features).apply(
        lambda feature: df.select([feature, target_column]).na.drop("any").count()
    ).astype(float)
    return out


def f_regression(target_column, features, df):
    """F-statistic between one numeric target column and many numeric columns of a Spark DataFrame

    F-statistic is actually obtained from the Pearson's correlation coefficient through the following formula:
    corr_coef ** 2 / (1 - corr_coef ** 2) * degrees_of_freedom
    where degrees_of_freedom = n_instances - 1.

    Parameters
    ----------
    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic between each column and the target column.
    """

    corr_coef = correlation(target_column=target_column, features=features, df=df)
    n = notna(target_column=target_column, features=features, df=df)

    deg_of_freedom = n - 2
    corr_coef_squared = corr_coef ** 2
    f = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    return f


def f_classif(target_column, features, df):
    groupby = df.replace(float('nan'), None).groupBy(target_column)

    avg = groupby.agg({feature: 'mean' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[4:-1], axis=1)

    var = groupby.agg({feature: 'var_pop' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[8:-1], axis=1)

    n = groupby.agg({feature: 'count' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[6:-1], axis=1)

    f = groupstats2fstat(avg=avg, var=var, n=n)
    f.name = target_column

    return f




def mrmr_regression(df, target_column, K, features=None, denominator='mean', only_same_domain=False,
                    return_scores=False, show_progress=True):
    """MRMR feature selection for a regression task

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    target_column: str
        Name of target column.

    K: int
        Number of features to select.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected: list of str
        List of selected features.
    """

    if features is None:
        features = get_numeric_features(df=df, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'target_column': target_column, 'features': features, 'df': df}
    redundancy_args = {'df': df}

    return mrmr_base(K=K, relevance_func=f_regression, redundancy_func=correlation,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)


