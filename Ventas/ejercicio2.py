import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('sakila.db')

df = pd.read_sql('''
    SELECT
        rental.rental_id, rental.rental_date, rental.return_date,
        customer.last_name AS apellido_cliente,
        store.store_id,
        city.city AS ciudad,
        film.title AS titulo_pelicula, film.rental_duration AS periodo_rentado,
        film.rental_rate AS precio_Alquiler, film.replacement_cost AS costo_pelicula,
        film.rating AS clasificacion
    FROM rental
    INNER JOIN customer ON rental.customer_id == customer.customer_id
    INNER JOIN inventory ON rental.inventory_id == inventory.inventory_id
    INNER JOIN store ON inventory.store_id == store.store_id
    INNER JOIN address ON store.address_id == address.address_id
    INNER JOIN city ON address.city_id == city.city_id
    INNER JOIN film ON inventory.film_id == film.film_id
    ;
''', conn, index_col='rental_id', parse_dates=['rental_date', 'return_date'])
print('Comprobacion de la Base')
print(df.head())
plt.figure()
df['periodo_rentado'].value_counts().plot(kind='bar', figsize=(14,6))
plt.savefig('bar periodo_rentado.png')
plt.close()
# Pelicula mas rentada
plt.figure()
df['precio_Alquiler'].value_counts().plot(kind='pie', figsize=(6,6))
plt.savefig('graf torta montos alquilados.png')
plt.close()
plt.figure()
df['precio_Alquiler'].value_counts().plot(kind='bar', figsize=(14,6))
plt.savefig('bar montos alquilados.png')
plt.close()

# Costos de pelicula
plt.figure()
df['costo_pelicula'].plot(kind='box', vert=False, figsize=(14,6))

plt.savefig('box costo_pelicula')
plt.close()
# Densidad Costo pelicula
plt.figure()

ax = df['costo_pelicula'].plot(kind='density', figsize=(14,6))
ax.axvline(df['costo_pelicula'].mean(), color='red')
ax.axvline(df['costo_pelicula'].median(), color='blue')
plt.savefig('density costo_pelicula')

plt.close()
# Precio total de peliculas
df['clasificacion'].value_counts()
# your code goes here
plt.figure()
df['clasificacion'].value_counts().plot(kind='bar', figsize=(14,6))

plt.savefig('bar clasificacion count')
plt.close()
#Relacion de costo por Clasificacion
plt.figure()
# your code goes here
df[['costo_pelicula', 'clasificacion']].boxplot(by='clasificacion', figsize=(14,6))

plt.savefig('boxplot clasificacion precio')
plt.close()
#Dias rentados
df['rental_days'] = df[['rental_date', 'return_date']].apply(lambda x: (x[1] - x[0]).days, axis=1)
#comprobacion
print(df['rental_days'].head())


#Analisis del promedio de tiempo rentado
df['rental_days'].mean()
# your code goes here
plt.figure()
ax = df['rental_days'].plot(kind='density', figsize=(14,6))
ax.axvline(df['rental_days'].mean(), color='red')
plt.savefig('rental_days')

plt.close()

#relacion tiempo alquilado y ganancia

# your code goes here
df['film_daily_rental_rate'] = df['precio_Alquiler'] / df['periodo_rentado']
#combacion
print('Comprobacion')
print(df['film_daily_rental_rate'].head())


#Analyze the distribution of film_daily_rental_rate
#Calculate the mean of film_daily_rental_rate.
#Show a density (KDE) of film_daily_rental_rate.
# your code goes here
df['film_daily_rental_rate'].mean()
# your code goes here
plt.figure()
ax = df['film_daily_rental_rate'].plot(kind='density', figsize=(14,6))
ax.axvline(df['film_daily_rental_rate'].mean(), color='red')

plt.savefig('flm density')
plt.close()
print('Lista 10 films menos dias rentada')

print(df.loc[df['film_daily_rental_rate'] == df['film_daily_rental_rate'].min()].head(10))


print('Lista 10 films mas dias rentada')
print(df.loc[df['film_daily_rental_rate'] == df['film_daily_rental_rate'].max()].head(10))


# Cantidad de Rentas por ciudad
df.loc[df['ciudad'] == 'Lethbridge'].shape[0]


# Cantidad de Rentas por ciudad segun la clasificacion
plt.figure()
df.loc[df['ciudad'] == 'Lethbridge', 'clasificacion'].value_counts()
df.loc[df['ciudad'] == 'Lethbridge', 'clasificacion'].value_counts().plot(kind='bar', figsize=(14,6))
plt.savefig('Rentas por ciudad')
plt.close()