import pandas as pd

data = pd.read_csv("Титаник.csv")

gender_info = data["Sex"].value_counts(dropna=False)
print(f'Количество мужчин в корабле: {gender_info["male"]}')

survived_percent = data["Survived"].value_counts(normalize=True)
print(f'Доля выживших пассажиров: {round(survived_percent[1] * 100, 2)}')

class_passenger_percent = data["Pclass"].value_counts(normalize=True)
print(f'Доля пассажиров путешествующих в 2 классе: {round(class_passenger_percent[2] * 100, 2)}')

print(f'Среднее значение возраста пассажирож: {data["Age"].mean()}')

print(f'Медианная значение возраста пассажирож: {data["Age"].median()}')

correlotion = data["SibSp"].corr(data['Parch'])
print(f'Коррелицая число братьев/сестер с числом родителей/детей: {round(correlotion, 2)}')
