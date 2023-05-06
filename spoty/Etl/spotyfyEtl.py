import sqlalchemy
import pandas as pd 
from sqlalchemy.orm import sessionmaker
import requests
import json
from datetime import datetime
import datetime
import sqlite3





DATABASE_LOCATION = "sqlite:///base.db"
USER_ID = "Agregar usuario Propio"
TOKEN = "BQAqoFTaNVLAjK417S092tdiiQblv__NzR6rA_5c5YDgixcEUAAPuUTFoYVNYAM5TfEQ0p0tlscwMTFAx75Kgea0UBqXurMthvYnN4mCFdr4D5XO6yqOmO9cc3jvxn2vmvOZ91E1cMOANduDhCSsokvpwXiVsjZH3ssYJDaHkwKSrZu2CldIYxc0n0wgSw4SUDHuoaTRH7PQnY8OB3vDdKMIEj0gI2TmFy6WFVsdM8quvJPwcAR7FJYVz8cE9_fICiedu7WEkoRO3KI99_n6i5Rxp_1MpSDdqXvC6h6dWQ2fsMmVEtwsYjDP0rijLsPk_7PZuvQ06l8OP0pvh-FDGwE1CA"


if __name__ == "__main__":

    # Extract part of the ETL process
 
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json",
        "Authorization" : "Bearer {token}".format(token=TOKEN)
    }
    
    # Convert time to Unix timestamp in miliseconds      
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    yesterday_unix_timestamp = int(yesterday.timestamp()) * 1000

    # Download all songs you've listened to "after yesterday", which means in the last 24 hours      
    r = requests.get("https://api.spotify.com/v1/me/player/recently-played?after={time}".format(time=yesterday_unix_timestamp), headers = headers)

    data = r.json()
    #print(data)
    song_names = []
    artist_names = []
    played_at_list = []
    timestamps = []

    for song in data["items"]:
        song_names.append(song["track"]["name"])
        artist_names.append(song["track"]["album"]["artists"][0]["name"])
        played_at_list.append(song["played_at"])
        timestamps.append(song["played_at"][0:10])
     
    # Prepare a dictionary in order to turn it into a pandas dataframe below       
    song_dict = {
        "song_name" : song_names,
        "artist_name": artist_names,
        "played_at" : played_at_list,
        "timestamp" : timestamps
    }

    song_df = pd.DataFrame(song_dict, columns = ["song_name", "artist_name", "played_at", "timestamp"])
    
#Validacion
    def check_if_valid_data(df: pd.DataFrame) -> bool:
    # Verificacion de que no este vacio
        if df.empty:
            print("No hay canciones descargadas. Execusion finalizada")
            return False 

    # Primary Key Check
        if pd.Series(df['played_at']).is_unique:
            pass
        else:
            raise Exception("Primary Key check no cumplido")

    # Check for nulls
        if df.isnull().values.any():
            raise Exception("Null values encontrados")

    # Check that all timestamps are of yesterday's date
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

        timestamps = df["timestamp"].tolist()
        #for timestamp in timestamps:
        #    if datetime.datetime.strptime(timestamp, '%Y-%m-%d') != yesterday:
        #        raise Exception("Una de las canciones no tiene fecha de ayer")

        return True
    if check_if_valid_data(song_df):
        print("Datos validados,procedamos con la carga")
    

    engine = sqlalchemy.create_engine(DATABASE_LOCATION)
    conn = sqlite3.connect('base.db')
    cursor = conn.cursor()

    sql_query = """
    CREATE TABLE IF NOT EXISTS my_played_tracks(
        song_name VARCHAR(200),
        artist_name VARCHAR(200),
        played_at VARCHAR(200),
        timestamp VARCHAR(200),
        CONSTRAINT primary_key_constraint PRIMARY KEY (played_at)
    )
    """

    cursor.execute(sql_query)
    print("Base de datos abierta")

    try:
        song_df.to_sql("base", engine, index=False, if_exists='append')
    except:
        print("Datos en la Base de datos")

    conn.close()
    print("Proceso finalizado")