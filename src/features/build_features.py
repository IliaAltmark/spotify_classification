"""
Author: Ilia Altmark
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def remove_outliers(df_train, df_val, df_test, cols):
    for col in cols:
        std = df_train[col].std()
        mean = df_train[col].mean()
        bottom_bound = max(mean - 3 * std, 0)
        top_bound = mean + 3 * std

        num_of_outliers = ((df_train[col] < bottom_bound) | (
                df_train[col] > top_bound)).sum()
        print(
            f'Ratio of outliers for column {col}: {num_of_outliers / len(df_train)}')

        if num_of_outliers / len(df_train) < 0.025:
            df_train = df_train[df_train[col].between(bottom_bound, top_bound)]
        else:
            df_train.loc[df_train[col] > top_bound, col] = top_bound
            df_train.loc[df_train[col] < bottom_bound, col] = bottom_bound

        df_val.loc[df_val[col] > top_bound, col] = top_bound
        df_val.loc[df_val[col] < bottom_bound, col] = bottom_bound

        df_test.loc[df_test[col] > top_bound, col] = top_bound
        df_test.loc[df_test[col] < bottom_bound, col] = bottom_bound

    return df_train, df_val, df_test


def main():
    df = pd.read_csv(
        'xxx',
        index_col=0)

    train, test = train_test_split(df,
                                   test_size=0.3,
                                   random_state=42)

    val, test = train_test_split(test,
                                 test_size=0.66,
                                 random_state=42)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("TRAIN Dataset: {}".format(train.shape))
    print("VAL Dataset: {}".format(val.shape))
    print("TEST Dataset: {}".format(test.shape))

    t_median = train['popularity'].median()

    print(f'The median is {t_median}')

    train['popularity_bin'] = (train['popularity'] > t_median).astype('int')
    val['popularity_bin'] = (val['popularity'] > t_median).astype('int')
    test['popularity_bin'] = (test['popularity'] > t_median).astype('int')

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    numerical = df.select_dtypes(include=['int64', 'float64']).columns.drop(
        ['key', 'mode', 'time_signature'])

    train[numerical] = imp.fit_transform(train[numerical])
    val[numerical] = imp.transform(val[numerical])
    test[numerical] = imp.transform(test[numerical])

    quantile_5 = train[train['popularity_bin'] == 1]['popularity'].quantile(
        0.05)
    print(f"The 5% quantile for popular tracks: {quantile_5}")

    quantile_5_sum_r = sum(
        train[train['popularity_bin'] == 1]['popularity'] <= quantile_5)
    print(f"Number of tracks that will be removed: {quantile_5_sum_r}")

    quantile_95 = train[train['popularity_bin'] == 0]['popularity'].quantile(
        0.95)
    print(f"The 95% quantile for not-popular tracks: {quantile_95}")

    quantile_95_sum_r = sum(
        train[train['popularity_bin'] == 0]['popularity'] >= quantile_95)
    print(f"Number of tracks that will be removed: {quantile_95_sum_r}")

    train = train[(train['popularity'] < quantile_95) | (
            train['popularity'] > quantile_5)].copy()

    train_len_before = len(train)

    train, val, test = remove_outliers(train, val, test,
                                       numerical.drop('loudness'))

    print(f'''Current ratio of rows (out of before the deletion): 
    train - {len(train) / train_len_before}''')

    features = train.select_dtypes(include=['int64', 'float64']).columns.drop(
        ['popularity', 'popularity_bin'])
    label = 'popularity_bin'

    X_train = train[features]
    y_train = train[label]

    X_val = val[features]
    y_val = val[label]

    X_test = test[features]
    y_test = test[label]

    X_train.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_train.csv')
    y_train.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_train.csv')

    X_val.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_val.csv')
    y_val.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_val.csv')

    X_test.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_test.csv')
    y_test.to_csv(
        '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_test.csv')


if __name__ == "__main__":
    main()
