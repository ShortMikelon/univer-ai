import pandas as pd
import time
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Загрузка данных из CSV
file_path = 'datasets/c10d1к.csv'  # Замените на путь к вашему файлу CSV
data = pd.read_csv(file_path, header=None, dtype='string')
data = data[0].apply(lambda x: x.split(','))  # Разделение элементов по запятой в каждой строке

# Преобразуем данные в формат, понятный алгоритмам
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data, columns=te.columns_)

# Установка диапазона поддержки
support_range = [x / 100 for x in range(2, 101, 2)]  # От 2% до 100% с шагом 2

# Списки для записи результатов
apriori_times = []
fp_growth_times = []

# Запуск алгоритмов по разным значениям min_support
for min_support in support_range:
    # Время выполнения Apriori
    start_apriori = time.time()
    apriori(df, min_support=min_support, use_colnames=True)
    end_apriori = time.time()
    apriori_times.append(end_apriori - start_apriori)

    # Время выполнения FP-Growth
    start_fp = time.time()
    fpgrowth(df, min_support=min_support, use_colnames=True)
    end_fp = time.time()
    fp_growth_times.append(end_fp - start_fp)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot([s * 100 for s in support_range], apriori_times, label='Apriori', marker='o')
plt.plot([s * 100 for s in support_range], fp_growth_times, label='FP-Growth', marker='o')

# Настройка осей и заголовков
plt.xlabel('Минимальная поддержка (%)')
plt.ylabel('Время выполнения (секунды)')
plt.title('Сравнение времени выполнения алгоритмов Apriori и FP-Growth')
plt.legend()
plt.grid(True)
plt.show()
