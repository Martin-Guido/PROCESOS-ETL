import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#Iniciamos el proceso de carga de datos
df = pd.read_csv('dataset.csv', header=0, dtype= {'STATE' : str, 'OVERALL_RANK' :  np.int32, 'AFFORDABILITY':np.int32,'CRIME':np.int32,'CULTURE':np.int32,'WEATHER':np.int32,'WELLNESS':np.int32})
#Almacenamos un Respaldo en nuesta base de datos
con=sqlite3.connect('dataset.db')
df.to_sql('datos originales',con,if_exists='replace', index=False)

# Defino un nuevo parametro que involucra las calificaciones
df['New_Rank'] = (1*df[	'AFFORDABILITY']+1*df[	'WEATHER']+df[	'CULTURE']+1.5*df[	'WELLNESS']-1.5*df[	'CRIME'])/3
#Reordeno los resultados
data=df[['STATE','New_Rank']].sort_values(by='New_Rank')
colors=df[	'CULTURE']
fig, ax = plt.subplots()

sc = ax.scatter(data['STATE'], data['New_Rank'], s=(df[	'AFFORDABILITY']*df[	'WEATHER']*df[	'WELLNESS'])/50, c=colors,edgecolors='face', alpha=0.5)
ax.tick_params(axis='x', rotation=75,labelsize=6)
cbar = fig.colorbar(sc)
cbar.set_label("Cultura")

plt.show()
