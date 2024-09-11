import pandas as pd
import re

def extract_first_name(full_name):
    match = re.search(r'\((.*?)\)', full_name)

    if match:
        return match.group(1).split()[0]
    
    parts = full_name.split(',')
    name_part = parts[1].strip().split('. ')[1]

    if ' ' in name_part:
        return name_part.split()[0]
    else:
        return name_part

data = pd.read_csv("Титаник.csv")

gender_info = data["Sex"].value_counts(dropna=False)
print(f'Количество мужчин в корабле: {gender_info["male"]}')

survived_percent = data["Survived"].value_counts(normalize=True)
print(f'Доля выживших пассажиров: {round(survived_percent[1] * 100, 2)}')

class_passenger_percent = data["Pclass"].value_counts(normalize=True)
print(f'Доля пассажиров путешествующих в 2 классе: {round(class_passenger_percent[2] * 100, 2)}')

print(f'Среднее значение возраста пассажирож: {round(data["Age"].mean(), 2)}')

print(f'Медианная значение возраста пассажирож: {data["Age"].median()}')

correlotion = data["SibSp"].corr(data['Parch'])
print(f'Коррелицая число братьев/сестер с числом родителей/детей: {round(correlotion, 2)}')

female_names = data[data["Sex"] == 'female']['Name']
female_names = female_names.apply(extract_first_name)

popular_female_name, popular_female_name_count = next(female_names.value_counts(sort=True).items())
print(f'Самый популярное женская имя: {popular_female_name}, количество: {popular_female_name_count}')
