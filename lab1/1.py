import re
import pandas as pd

# Извлекает первое имя из полного имени
def extract_first_name(full_name):
    # Если имя указано в скобках
    if match := re.search(r'\((.*?)\)', full_name):
        return match.group(1).split()[0]
    
    # Если формат имени "Фамилия, Имя Отчество"
    name_part = full_name.split(',')[1].strip().split('. ')[1]
    return name_part.split()[0]

# Загрузка данных
data = pd.read_csv("Титаник.csv")

# Количество мужчин
male_count = data["Sex"].value_counts(dropna=False).get("male", 0)
print(f'Количество мужчин на корабле: {male_count}')

# Доля выживших
survival_rate = round(data["Survived"].mean() * 100, 2)
print(f'Доля выживших пассажиров: {survival_rate}%')

# Доля пассажиров 2 класса
second_class_rate = round(data["Pclass"].value_counts(normalize=True).get(2, 0) * 100, 2)
print(f'Доля пассажиров 2 класса: {second_class_rate}%')

# Средний возраст пассажиров
average_age = round(data["Age"].mean(), 2)
print(f'Средний возраст пассажиров: {average_age}')

# Медианный возраст пассажиров
median_age = data["Age"].median()
print(f'Медианный возраст пассажиров: {median_age}')

# Корреляция между количеством братьев/сестер и родителей/детей
sibsp_parch_corr = round(data["SibSp"].corr(data["Parch"]), 2)
print(f'Корреляция числа братьев/сестер с числом родителей/детей: {sibsp_parch_corr}')

# Самое популярное женское имя
female_names = data[data["Sex"] == "female"]["Name"].apply(extract_first_name)
popular_name, name_count = female_names.value_counts().idxmax(), female_names.value_counts().max()
print(f'Самое популярное женское имя: {popular_name}, количество: {name_count}')
