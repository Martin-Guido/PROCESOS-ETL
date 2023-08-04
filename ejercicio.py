import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sales = pd.read_csv('sales_data.csv',
    parse_dates=['Date'])
print('Comprobacion ventas')
print(sales.head())


# Clientes por edad


plt.figure()
sales['Customer_Age'].plot(kind='kde', figsize=(14,6))
plt.savefig('Clientes edad.png')
plt.close()

plt.figure()
sales['Customer_Age'].plot(kind='box', vert=False, figsize=(14,6))
plt.savefig('Clientes edad box.png')
plt.close()

# Histograma de la Cantidad de ordenes
plt.figure()
sales['Order_Quantity'].plot(kind='hist', bins=30, figsize=(14,6))
plt.savefig('hist cantidad de Ordenes.png')
plt.close()

plt.figure()
sales['Order_Quantity'].plot(kind='box', figsize=(14,6))
plt.savefig('cantidad de Ordenes box.png')
#Cantidad de ventas por anios
plt.close()

#GRaf torta de cantidad de ventas anuales
sales['Year'].value_counts().plot(kind='pie', figsize=(6,6))
# Cantodad de Ventas por mes

sales['Month'].value_counts().plot(kind='bar', figsize=(14,6))


plt.figure()
# Ventas por pais
sales['Country'].value_counts().plot(kind='bar', figsize=(14,6))
plt.savefig('Bar pais.png')
plt.close()

#Crear lista de productos
sales.loc[:, 'Product'].unique()

sales['Product'].unique()
# GRafica de los 10 mas vendidos
plt.figure()
# your code goes here
sales['Product'].value_counts().head(10).plot(kind='bar', figsize=(14,6))

plt.savefig('Productos top 10.png')
plt.close()

#Relacion entre precio unitario y costo unitario
plt.figure()
sales.plot(kind='scatter', x='Unit_Cost', y='Unit_Price', figsize=(6,6))
plt.savefig('relacion precio costo.png')
plt.close()

# Relacion de ordenes
plt.figure()

sales.plot(kind='scatter', x='Order_Quantity', y='Profit', figsize=(6,6))
plt.close()

plt.savefig('relacion ordenes.png')


plt.figure()

sales[['Profit', 'Country']].boxplot(by='Country', figsize=(10,6))
plt.savefig('boxplot por pais.png')
plt.close()

# Relacion edad de los clientes con los paises.
plt.figure()
# your code goes here
sales[['Customer_Age', 'Country']].boxplot(by='Country', figsize=(10,6))
plt.savefig('Edad por pais.png')
plt.close()

#Calculamos la fecha de ventas
sales['Calculated_Date'] = sales[['Year', 'Month', 'Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)
print('Comprobacion de la fecha de ventas')
print(sales['Calculated_Date'].head())
# Lo transformamos en formato fecha
sales['Calculated_Date'] = pd.to_datetime(sales['Calculated_Date'])
print('nueva Comprobacion de la fecha de ventas')
print(sales['Calculated_Date'].head())

#grafico de la cantidad de ventas por fecha
plt.figure()
# your code goes here
sales['Calculated_Date'].value_counts().plot(kind='line', figsize=(14,6))
#incremento general en $50
plt.savefig('vntas en l tiempo.png')

sales['Revenue'] = sales['Revenue'] + 50

sales['Revenue'] += 50

#Cantidad de ventas por pais ejemplo canada francia
sales.loc[(sales['Country'] == 'Canada') | (sales['Country'] == 'France')].shape[0]


#Cantidad de poducto por pais ejemplo bicis en canada
sales.loc[(sales['Country'] == 'Canada') & (sales['Sub_Category'] == 'Bike Racks')].shape[0]

#cantidad de ventas por region de un pais
france_states = sales.loc[sales['Country'] == 'France', 'State'].value_counts()


plt.figure()
france_states.plot(kind='bar', figsize=(14,6))
plt.savefig('bara frances.png')

#Ventas por categoria de productos
sales['Product_Category'].value_counts()

sales['Product_Category'].value_counts().plot(kind='pie', figsize=(6,6))
#Cantidad de ventas por productos
accessories = sales.loc[sales['Product_Category'] == 'Accessories', 'Sub_Category'].value_counts()

plt.figure()
# your code goes here
accessories.plot(kind='bar', figsize=(14,6))
plt.savefig('bara accessories.png')

#Cantidad de ventas por producto ejemplo bicis
bikes = sales.loc[sales['Product_Category'] == 'Bikes', 'Sub_Category'].value_counts()


plt.figure()
# your code goes here
bikes.plot(kind='pie', figsize=(6,6))
plt.savefig('bci torta.png')
plt.figure()
#Ventas por genero
sales['Customer_Gender'].value_counts()
sales['Customer_Gender'].value_counts().plot(kind='bar')
plt.savefig('generos ventas.png')