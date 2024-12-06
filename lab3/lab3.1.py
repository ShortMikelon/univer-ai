import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Титаник.csv")

data_selected = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

data_selected.loc[:, 'Sex'] = data_selected['Sex'].map({'male': 0, 'female': 1})

data_selected = data_selected[~data_selected.isna().any(axis=1)]

X = data_selected[['Pclass', 'Fare', 'Age', 'Sex']]
y = data_selected['Survived']

clf = DecisionTreeClassifier(random_state=100)
clf.fit(X, y)

feature_importance = clf.feature_importances_
important_features = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)

print(important_features.head(2))