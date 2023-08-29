import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Wczytanie danych z pliku CSV
data = pd.read_csv('PRSA_data_2010.1.1-2014.12.csv', sep=",", encoding='utf-8')

data.info()

# Wyczyszczenie danych
# Uzupełnienie brakujących wartości w kolumnie "year", "month", "day" (forward propagation)
data['year'].fillna(method='ffill', inplace=True)
data['month'].fillna(method='ffill', inplace=True)
data['day'].fillna(method='ffill', inplace=True)
data['hour'].interpolate(method='linear', inplace=True)

data['DEWP'].interpolate(method='linear', inplace=True)
data['TEMP'].interpolate(method='linear', inplace=True)
data['PRES'].interpolate(method='linear', inplace=True)
# usunięcie wierszy dla których w kolumnie pm2.5 występuje NAN
data.dropna(subset=['pm2.5'], inplace=True)

# Tworzenie kolumny Timestamp na podstawie kolumn (Rok, Miesiąc, Dzień i Godzina)
data['Timestamp'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

data = data.drop('No', axis=1)
data = data.drop('year', axis=1)
data = data.drop('month', axis=1)
data = data.drop('day', axis=1)
data = data.drop('hour', axis=1)
data.head()

data.info()

# Wykres 1: Zmiany parametrów w czasie
values = data.values
valuesCopy = values.copy()

col = [0, 1, 2, 3, 5, 6, 7]

fig, axs = plt.subplots(len(col), 1, figsize=(20, len(col) * 2), sharex=True)
fig.suptitle("Zmiany parametrów w czasie", fontsize=16)

for i, feature in enumerate(col):
    axs[i].plot(data['Timestamp'], data.iloc[:, feature])
    axs[i].set_title(data.columns[feature], loc='left')
plt.tight_layout()
#plt.show()

# Wykres 2: pudełkowy (box plot) - rozkład poziomu PM2.5 w różnych miesiącach
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.to_datetime(data['Timestamp']).dt.month, y='pm2.5', data=data)
plt.xlabel('Miesiąc')
plt.ylabel('Poziom PM2.5 (ug/m3)')
plt.title('Rozkład poziomu PM2.5 w różnych miesiącach')
# plt.savefig('nieoczyszcone/miesieczny_rozkladPM25.png')
#plt.show()

print(f"data:\n", data)

# Zmiana zmiennych kategorycznych na numeryczne
le = LabelEncoder()
data['cbwd'] = le.fit_transform(data['cbwd'])

col_float = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
data[col_float] = data[col_float].round(2).astype('float32')
print(f"encoded data:\n", data.head())

data.info()

# dodanie kolumny PM2.5 z następnego zapisu(godzina do przodu)
"""dodaje kolumnę z następnego zapisu, aby sprawdzić,
     czy PM2.5 uzyskuje wyższą korelacje z attrybutami
    z aktualnej godziny, czy godziny poprzedniej  """
pm_1h_after = data['pm2.5'].shift(-1)  # sprawdzenie czy pm2.5 jest zależne od parametrów z poprzedniego zapisu
data['pm_1h_after'] = pm_1h_after
data.dropna(subset=['pm_1h_after'], inplace=True)

# Wykres 3: korelacje - macierz korelacji między atrybutami jakości powietrza
"""sprawdzam czy korelacje między poszczególnymi
         atrybutami nie są zbyt duże,
   aby wykluczyć ewentualnie te, które są zbędne"""
corr_matrix = data[['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']].corr()
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji między atrybutami jakości powietrza')
# plt.savefig('oczyszczone/korelacje_atrybutow.png')
#plt.show()

# Wykres 4: korelacje - macierz korelacji między PM2.5 a atrybutami jakości powietrza
# wymnożenie *100 pozwala na odczytanie wartości w procentach
correlation = abs(data.corr()['pm2.5'].drop('pm2.5') * 100).sort_values(ascending=False)
correlation.plot.bar(figsize=(12, 6))
plt.title('Macierz korelacji między PM2.5 a atrybutami jakości powietrza')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja (%)')
for i, val in enumerate(correlation):
    plt.text(i, val, f'{val:.2f}', horizontalalignment='center', verticalalignment='bottom')

# plt.savefig('oczyszczone/korelacje_PM25.png')
#plt.show()

# Wykres 5: korelacje - macierz korelacji między PM2.5 (pm_1h_after) a atrybutami jakości powietrza
# wymnożenie *100 pozwala na odczytanie wartości w procentach
correlation = abs(data.corr()['pm_1h_after'].drop(['pm_1h_after', 'pm2.5']) * 100).sort_values(ascending=False)
correlation.plot.bar(figsize=(12, 6))
plt.title('Macierz korelacji między PM2.5 (pm_1h_after) a atrybutami jakości powietrza')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja (%)')
for i, val in enumerate(correlation):
    plt.text(i, val, f'{val:.2f}', horizontalalignment='center', verticalalignment='bottom')

# plt.savefig('oczyszczone/korelacje_PM25.png')
#plt.show()

# Wykres 6: korelacje - macierz korelacji między PM2.5 a atrybutami jakości powietrza
# wymnożenie *100 pozwala na odczytanie wartości w procentach
correlation = abs(data.corr()['pm2.5'].drop('pm2.5') * 100).sort_values(ascending=False)
correlation.plot.bar(figsize=(12, 6))
plt.title('Macierz korelacji między PM2.5 a atrybutami jakości powietrza')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja (%)')
# plt.savefig('oczyszczone/korelacje_PM25.png')
#plt.show()

# Normalizacja danych numerycznych
scaler = MinMaxScaler()
col_stand = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
for i in data[col_stand]:
    data[i] = scaler.fit_transform(data[[i]])
print(f"normalized data:\n",
      data.head())

# Wykluczanie niektórych kolumn z analizy ()
exclude_filter = ~data.columns.isin(['Timestamp'])

# przygotowanie danych do podziału
X = data[['pm2.5', 'TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']]  # dane wejściowe
y = data['pm_1h_after']  # dane wyjściowe

train_years = 3  # liczba lat zawartych w zestawie treningowym
n_train_hours = train_years * 365 * 24  # liczba wierszy dla zestawu treningowego

# Podział danych na zbiór treningowy i testowy
X_train = X[:n_train_hours]  # dane wejściowe dla zestawu treningowego - pierwsze n_train_hours
y_train = y[:n_train_hours]  # dane wyjściowe dla zestawu treningowego - pierwsze n_train_hours
X_test = X[n_train_hours:]
y_test = y[n_train_hours:]

data.info()
data.head()

# regresja liniowa
linReg = LinearRegression().fit(X_train, y_train)
# to zwraca R^2
linReg_score_train = linReg.score(X_train, y_train)
linReg_score_test = linReg.score(X_test, y_test)

y_predict = linReg.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred=y_predict)
reglin_rmse_test = np.sqrt(mse)
print(f"linear RMSE:\n", reglin_rmse_test)
print(f"linear R^2:\n", linReg_score_train)

# Regresja liniowa LASSO
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

lasso_score_train = lasso.score(X_train, y_train)
lasso_score_test = lasso.score(X_test, y_test)

y_predict = lasso.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred=y_predict)
lasso_rmse_test = np.sqrt(mse)
print(f"lasso RMSE:\n", lasso_rmse_test)
print(f"lasso R^2:\n", lasso_score_train)

# Regresja wielomianowa stopnia 3/2/..
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

poly_score_train = lin_reg.score(X_train_poly, y_train)
poly_score_test = lin_reg.score(X_test_poly, y_test)

y_predict = lin_reg.predict(X_test_poly)
mse = mean_squared_error(y_true=y_test, y_pred=y_predict)
poly_rmse_test = np.sqrt(mse)
print(f"poly RMSE:\n", poly_rmse_test)
print(f"poly R^2:\n", poly_score_train)

# Regresja z wykorzystaniem k-NN
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(X_train, y_train)

knn_score_train = knn.score(X_train, y_train)
knn_score_test = knn.score(X_test, y_test)

y_predict = knn.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred=y_predict)
knn_rmse_test = np.sqrt(mse)
print(f"knn RMSE:\n", knn_rmse_test)
print(f"knn R^2:\n", knn_score_train)

# Regresja z wykorzystaniem drzewa decyzyjnego
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

tree_score_train = tree.score(X_train, y_train)
tree_score_test = tree.score(X_test, y_test)

y_predict = tree.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred=y_predict)
tree_rmse_test = np.sqrt(mse)
print(f"tree RMSE:\n", tree_rmse_test)
print(f"tree R^2:\n", tree_score_train)

# Dane do wyniki
train_scores = [linReg_score_train, lasso_score_train, poly_score_train, knn_score_train, tree_score_train]
rmse_test = [reglin_rmse_test, lasso_rmse_test, poly_rmse_test, knn_rmse_test, tree_rmse_test]
test_scores = [linReg_score_test, lasso_score_test, poly_score_test, knn_score_test, tree_score_test]
models = ['Linear Regression', 'Lasso', 'Polynomial Regression', 'k-NN', 'Decision Tree']

# Wykres z wynikami
plt.figure(figsize=(10, 5))
plt.plot(models, train_scores, label='Train Score')
plt.plot(models, test_scores, label='Test Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.legend()
# plt.savefig('porownanie/porownanie modeli.png')
# plt.show()

# Tabela wyników
results = pd.DataFrame({'Model': models,
                        'RMSE': rmse_test,
                        'Train Score': train_scores,
                        'Test Score': test_scores})
print(results)

# Wykres predykcji Lasso
data['predictions'] = lasso.predict(X)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# Tworzenie wcześniej usuniętych kolumn 'Year' i 'Month' na podstawie kolumny 'Timestamp'
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data.groupby(['Year', 'Month']).mean().plot(y=['pm2.5', 'predictions'], figsize=(15, 5), grid=True)
plt.legend()
plt.xlabel('Data')
plt.ylabel('pm2.5')
plt.title('prognoza PM2.5')
# plt.show()

# Wykres predykcji Regresji Liniowej
data['predictions'] = linReg.predict(X)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# Tworzenie wcześniej usuniętych kolumn 'Year' i 'Month' na podstawie kolumny 'Timestamp'
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data.groupby(['Year', 'Month']).mean().plot(y=['pm2.5', 'predictions'], figsize=(15, 5), grid=True)
plt.legend()
plt.xlabel('Data')
plt.ylabel('pm2.5')
plt.title('prognoza PM2.5')
plt.show()
