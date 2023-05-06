import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import datetime as dt
import squarify
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from operator import attrgetter
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

#Iniciamos el proceso de carga de datos
df = pd.read_csv("data.csv", header=0, dtype= {'CustomerID' : str, 'InvoiveID' :  str}, encoding='unicode_escape',
                    parse_dates=["InvoiceDate"], infer_datetime_format=True)
#Almacenamos un espaldo en nuesta base de datos
con=sqlite3.connect('data.db')
df.to_sql('datos originales',con,if_exists='replace', index=False)




world_map = df[['CustomerID', 'InvoiceNo', 'Country']
              ].groupby(['CustomerID', 'InvoiceNo', 'Country']
                       ).count().reset_index(drop = False)
countries = world_map["Country"].value_counts()

data = dict(type='choropleth',
            locations = countries.index,
            locationmode = 'country names',
            z = countries,
            text = countries.index,
            colorbar = {'title':'Ordenes'},
            colorscale='Viridis',
            reversescale = False)

layout = dict(title={'text': "Numero de Ordenes por Pais",
                     'y':0.9,
                     'x':0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'},
              geo = dict(resolution = 50,
                         showocean = True,
                         oceancolor = "LightBlue",
                         showland = True,
                         landcolor = "whitesmoke",
                         showframe = True),
             template = 'plotly_white',
             height = 600,
             width = 1000)

choromap = go.Figure(data = [data], layout = layout)
pyo.plot(choromap, filename='map.html')   

def desc_stats(dataframe):
    desc_df = pd.DataFrame(index= dataframe.columns, 
                           columns= dataframe.describe().T.columns,
                           data= dataframe.describe().T)
    
    f,ax = plt.subplots(figsize=(20,
                                 desc_df.shape[0] * 1))
    sns.heatmap(desc_df,
                annot = True,
                cmap = "Greens",
                fmt = '.2f',
                ax = ax,
                linecolor = 'white',
                linewidths = 1.1,
                cbar = True,
                annot_kws = {"size": 12})
    plt.xticks(size = 18)
    plt.yticks(size = 14,
               rotation = 0)
    plt.title("Estadistica Descriptiva", size = 16)
    plt.plot()
    plt.savefig('Estadisticadescriptiva.png')
           
desc_stats(df.select_dtypes(include = [float, int]))
df.plot.box()
plt.savefig('Grafico Box')

def filtrado_datos_atipicos(dataframe, col, q1=0.25, q3=0.75):
    '''
        Realiza el Capping en los valores atípicos: 
        que consiste en sustituir los valores atípicos por los umbrales inferior y superior.
        
        Args:

        dataframe -> contiene todos los datos
        col -> es el col del dataframe para el que estamos realizando el capping
        q1 -> es el primer cuantil con valor por defecto
        q3 -> es el tercer cuantil 

        devolver:

        dataframe 
    '''
    d = dataframe.copy()
    quantile1 = d[col].quantile(q1)
    quantile3 = d[col].quantile(q3)
    iqr = quantile3 - quantile1
    lower = quantile1 - 1.5*iqr
    upper = quantile3 + 1.5*iqr

    d.loc[(d[col] < lower), col] = lower
    d.loc[(d[col] > upper), col] = upper

    return d

def preproceso_comercial(dataframe):
    '''
        Realiza algunas Limpiezas en el dataframe mediante:
        * eliminando los registros que contienen valores nulos
        * eliminando los registros de pedidos cancelados
        * Eliminando los registros con valores negativos de cantidad
        * manejo de valores atípicos
        * creando una columna de precio total a partir de la cantidad y el precio unitario.
    '''
    df_ = dataframe.copy()
    
    #Valores perdidos
    df_ = df_.dropna()
    
    #Ordenes canceladas y Cantidades
    df_ = df_[~df_['InvoiceNo'].str.contains('C', na = False)]
    df_ = df_[df_['Quantity'] > 0]
    
    #Remplazando valores outsider
    df_ = filtrado_datos_atipicos(df_, "Quantity")
    df_ = filtrado_datos_atipicos(df_, "UnitPrice")
    
    #T Precio Total
    df_["TotalPrice"] = df_["Quantity"] * df_["UnitPrice"]
    
    return df_
# LLamamos la  funcion 
df = preproceso_comercial(df)

# Repetimos el proceso de estadistica descriptiva con los datos procesados
desc_stats(df.select_dtypes(include = [float,int]))

df.to_sql('Preproceso comercial',con,if_exists='replace', index=False)

# selecting as a current date 
today_date = dt.datetime(2011,12,11) 

# Evaluamos: recurencia, frecuencia y valores monetarios de los clientes
rfm = df.groupby("CustomerID").agg({'InvoiceDate' : lambda x : (today_date - x.max()).days,
                                    'InvoiceNo' : lambda x: x.nunique(),
                                    'TotalPrice' : lambda x: x.sum()})
rfm.columns = ["Recency","Frequency","Monetary"]
rfm["Monetary"] = rfm["Monetary"][rfm["Monetary"] > 0] 
rfm = rfm.reset_index()

rfm.to_sql('datos evaluados',con,if_exists='replace', index=False)

def Obtencion_Resultados_rfm(dataframe):
    
    df_ = dataframe.copy()
    df_['recency_score'] = pd.qcut(df_['Recency'],5,labels=[5, 4, 3, 2, 1])
    df_['frequency_score'] = pd.qcut(df_['Frequency'].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
    df_['monetary_score'] = pd.qcut(df_['Monetary'], 5, labels = [1, 2, 3, 4, 5])
    df_['RFM_SCORE'] = (df_['recency_score'].astype(str) + df_['frequency_score'].astype(str))
    
    return df_

rfm = Obtencion_Resultados_rfm(rfm)

seg_map = {r'[1-2][1-2]': 'Hibernando',
           r'[1-2][3-4]': 'En riesgo',
           r'[1-2]5': 'No Perder',
           r'3[1-2]': 'Dejarlos Descansar',
           r'33': 'Necesitan Atencion',
           r'[3-4][4-5]': 'Clientes Leales',
           r'41': 'Prometedores',
           r'51': 'Nuevos Clientes',
           r'[4-5][2-3]': 'Potencial Leales',
           r'5[4-5]': 'champions'}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)


segments = rfm['segment'].value_counts().sort_values(ascending = False)
plt.figure()
plt.title("Mapa de segmentacion", fontsize = 20)
plt.xlabel('Frecuencia', fontsize = 18)
plt.ylabel('Recurencia', fontsize = 18)
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 7)
squarify.plot(sizes=segments,
              label=[label for label in seg_map.values()],
              color=['#AFB6B5', '#F0819A', '#926717', '#F0F081', '#81D5F0',
                     '#C78BE5', '#748E80', '#FAAF3A', '#7B8FE4', '#86E8C0'],
              pad = False,
              bar_kwargs = {'alpha': 0.5},
              text_kwargs = {'fontsize':10})

plt.plot()
plt.savefig('Mapa de segmentacion')


x = rfm[["recency_score","frequency_score"]]
labels = rfm["segment"]
print("Numero de Observaciones : ", len(rfm))
print("Numero de Segmentos : ", len(labels.unique()))
print("Silhouette Score : ", round(silhouette_score(x, labels), 3))
print("Calinski Harabasz Score : ", round(calinski_harabasz_score(x, labels), 3))
print("Davies Bouldin Score :", round(davies_bouldin_score(x, labels), 3))
plt.figure(figsize=(10,10))

rfm[["Recency","Frequency","Monetary","segment"]].groupby("segment").agg({'mean', 'min', 'max', 'std'})

plt.pie(rfm["segment"].value_counts(), labels=rfm["segment"].value_counts().index, 
            autopct="%.0f%%", colors= sns.color_palette("bright"))
plt.plot()
plt.savefig('grafico de torta')

plt.figure(figsize=(14,8))
sns.scatterplot(data=rfm, x="Recency", y = "Frequency", hue="segment")
plt.title("Frecuencia por segmentos")
plt.legend(title="Segmentos", title_fontsize=14)
plt.plot()
plt.savefig('Frecuencia por segmentos')






def CohortAnalysis(dataframe):
    plt.figure()
    data = dataframe.copy()
    data = data[['CustomerID', 'InvoiceNo', 'InvoiceDate']].drop_duplicates()
    data['order_month'] = data['InvoiceDate'].dt.to_period('M')
    data['cohort'] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    cohort_data = data.groupby(['cohort', 'order_month']).agg(n_customers=('CustomerID', 'nunique')).reset_index(drop=False)
    cohort_data['period_number'] = (cohort_data.order_month - cohort_data.cohort).apply(attrgetter('n'))
    cohort_pivot = cohort_data.pivot_table(index = 'cohort',
                                           columns = 'period_number',
                                           values = 'n_customers')
    cohort_size = cohort_pivot.iloc[:,0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)

    # visualizing the retention matrix
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 8),
                        sharey=True,
                        gridspec_kw={'width_ratios': [1, 11]})
        sns.heatmap(retention_matrix, 
                    mask = retention_matrix.isnull(), 
                    annot = True,
                    cbar = False,
                    fmt='.0%', 
                    cmap='coolwarm', ax=ax[1])
        ax[1].set_title('Estudio de Cohortes mensuales: Clientes conservados', fontsize=14)
        ax[1].set(xlabel=' Periodo',
                  ylabel='')
        white_cmap = mcolors.ListedColormap(['white'])
        sns.heatmap(pd.DataFrame(cohort_size).rename(columns={0: 'Tamano Cohorte'}), 
                    annot=True, 
                    cbar = False,
                    fmt='g',
                    cmap=white_cmap,
                    ax=ax[0])
        fig.tight_layout()
        plt.plot()
        plt.savefig('Estudio de cohortes')

 
# Generamos la grafica de retencion de clientes     
CohortAnalysis(df)


# búsqueda de clientes, recurrencia, monetaria, frecuencia y duración total.
cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (today_date - x.min()).days], 
                                        'InvoiceNo': lambda x: x.nunique(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']


cltv_df.to_sql('Encontrados',con,if_exists='replace', index=False)




#Valor medio de los pedidos
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

#Recurencia
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

#Frecuencia
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]


bgf = BetaGeoFitter(penalizer_coef=0.0001) # for ovoiding overfitting
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# Generamos la grafica de ventas  Número esperado de transacciones Para el Top 10 de Clientes
plt.figure()



cf1=bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"]).sort_values(ascending = False).head(10).to_frame("1erSemana").reset_index()
cf2=bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"]).sort_values(ascending = False).head(10).to_frame("4taSemana").reset_index()

markers = {"1erSemana": "s", "4taSemana": "X"}

sns.scatterplot(data=cf1,x="CustomerID", y = "1erSemana",label='1erSemana')
sns.scatterplot(data=cf2,x="CustomerID", y = "4taSemana",label='4taSemana')
plt.title("Número esperado de transacciones")
plt.legend(title="Semanas")
plt.ylabel('Número esperado de transacciones')
plt.savefig('Número esperado de transacciones')


















