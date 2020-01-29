import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

wine_data = pd.read_csv("data\\training.csv", index_col=0)
wine_data_val = pd.read_csv("data\\validation.csv", index_col=0)

# Get rid of outliers
for cols in wine_data.columns:
    Q1 = wine_data[cols].quantile(0.25)
    Q3 = wine_data[cols].quantile(0.75)
    IQR = Q3 - Q1

    filts = (wine_data[cols] >= Q1 - 1.5 * IQR) & (wine_data[cols] <= Q3 + 1.5 * IQR)
    wine_data = wine_data.loc[filts]

X_train = wine_data.iloc[:, wine_data.columns != "quality"]
y_train = wine_data["quality"]
X_test = wine_data_val.iloc[:, wine_data_val.columns != "quality"]

# Scaling did not improve
# scal = StandardScaler()
# X = pd.DataFrame(scal.fit_transform(X), columns=X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)

# Baseline model
rf_baseline = RandomForestClassifier(n_estimators=1750, random_state=1)
rf_baseline.fit(X_train, y_train)

preds = rf_baseline.predict(X_test)

X_test["quality"] = preds

X_test.to_csv("DanielMacias_wine_preds.csv")

# Creation of .csv
wine_data = pd.read_csv("data\\training.csv", index_col=0)

# Get rid of outliers
for cols in wine_data.columns:
    Q1 = wine_data[cols].quantile(0.25)
    Q3 = wine_data[cols].quantile(0.75)
    IQR = Q3 - Q1

    filts = (wine_data[cols] >= Q1 - 1.5 * IQR) & (wine_data[cols] <= Q3 + 1.5 * IQR)
    wine_data = wine_data.loc[filts]

wine_data = pd.concat([wine_data, wine_data_val])
wine_data = wine_data.loc[:, wine_data.columns != "quality"]

rfc = RandomForestClassifier()
rfc.fit(wine_data)

# I tried but did not improve:
# Up-sample minority class
wd_X_train, wd_X_test, wd_y_train, wd_y_test = train_test_split(X, y, random_state=1, test_size=0.25)

train = pd.concat([wd_X_train, wd_y_train], axis=1)
wd_train_majority = train[train.quality.isin([5, 6, 7])]
wd_train_minority = train[~train.quality.isin([5, 6, 7])]

wd_minority_upsampled = resample(wd_train_minority,
                                 replace=True,      # sample with replacement
                                 n_samples=wd_train_majority.shape[0],     # to match majority class
                                 random_state=1)  # reproducible results

wd_train = pd.concat([wd_minority_upsampled, wd_train_majority])

wd_X_train = wd_train.iloc[:, wd_train.columns != "quality"]
wd_y_train = wd_train["quality"]

# Baseline model
rf_baseline = RandomForestClassifier(random_state=1)
rf_baseline.fit(wd_X_train, wd_y_train)
accuracy_score(rf_baseline.predict(wd_X_test), wd_y_test)
cohen_kappa_score(rf_baseline.predict(wd_X_test), wd_y_test)


