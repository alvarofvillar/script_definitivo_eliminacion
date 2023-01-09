# Título: Función de Eliminación de Incidencias en Jobs de Anomalías en Series Temporales entrenados con el módulo de Machine Learning de Elastic Stack. 
# Autor: Álvaro Fernández Villar
# Puesto: Data Scientist Pro. Caixabank Tech
#
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Descripcion: La ejecución del script permite realizar una conexión con un servidor de Elastic en el que haya un job de detección de anomalías en series temporales
#              ejecutandose y eliminar un periodo de aprendizaje correspondiente a una incidencia. Para ello es necesario introducir en la función como inputs la
#              fecha de inicio y final de la incidencia, el nombre del job de Elastic sobre el cual se quiere realizar la eliminacion del aprendizaje de la incidencia 
#              la variable detector del job, es decir, el nombre de la variable sobre la que se lleva a cabo la predicción en la serie temporal, el nombre del índice 
#              en el que se encuentran los datos sobre el cual se lleva a cabo el job de detección de anomalías y, finalmente, los datos que permiten la conexión con 
#              el servidor de Elastic, es decir, usuario, contraseña y host. Estos datos se introducen por medio de un fichero en formato .json. La función genera un fichero con la fecha de ejecución y los resultados del script. En el caso
#              de la existencia de errores manejables por la función, estos aparecerán en el fichero de salida.
#
# INPUTS DEL JSON:
# fecha_inicio_incidencia: Variable que indica la fecha de inicio de la incidencia según timezone para España (no UTC). String. Ejemplo: "2022-03-11 23:50:00"
# fecha_fin_incidencia:    Variable que indica la fecha de fin de la incidencia según timezone para España (no UTC). String. Ejemplo: "2022-03-13 00:10:00"
# jobname:                 Variable que indica el nombre del job de Elastic. String. Ejemplo: "job_pruebas_anomalias_deteccion"
# variable_name_predict:   Variable que indica el nombre de la variable sobre la que se realiza la prediccion. String. Ejemplo: "request_number"
# partition_field:         Variable que indica el nombre del campo por el que se realiza la particion en el Job de deteccion de anomalías. 
# index_data:              Variable que indica el nombre del índice sobre el que se lleva el Job de deteccion de Elastic. String. Ejemplo: "prueba_index"
# elastic_user:            Variable que indica el nombre del usuario que tiene acceso al clúster de Elastic. String. Ejemplo: "elastic"
# elastic_password:        Variable que indica la contraseña vinculada al usuario anterior que tiene acceso al clúster de Elastic. String. Ejemplo: "Elastic"
# elastic_host:            Variable que indica la dirección del host en el que esta alojado Elastic. String. Ejemplo: "localhost:9201"
#
# OUTPUTS: 
#   Model_Delete_Anomalies_Elastic_Function_Log_Result: Fichero de Log con resultados de la ejecución de la función y los errores, en el caso de existencia de los mismos. 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import elasticsearch
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import client
from datetime import datetime
from time import sleep
from elasticsearch.client import IndicesClient
import eland as ed
import pandas as pd
from elasticsearch.client import IngestClient
import json
import os
from pytz import timezone
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------------LECTURA DE FICHERO------------------------------------------------------------------
# Fichero de datos
file_input_Json = "Input_Delete_Anomalies_Elastic_partition.json"
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Manejo de errores de fechas
count_error = 0
# Manejo de errores de creacion de indices
count_error_2 = 0

# Lectura del fichero
try:
  file_input_Json = open(file_input_Json)
except:
  print("Fichero en formato incorrecto. El script finalizará. Error en linea 56.")
  count_error = count_error + 1
else:
  data = json.load(file_input_Json)
  fecha_inicio_incidencia = data["fecha_inicio_incidencia"]
  fecha_fin_incidencia = data["fecha_fin_incidencia"]
  diferencia_fechas_horas_max = data["diferencia_fechas_horas_max"]
  jobname = data["jobname"]
  variable_name_predict = data["variable_name_predict"]
  partition_field = data["partition_field"]
  index_data = data["index_data"]
  ELASTIC_USER = data["elastic_user"]
  ELASTIC_PASSWORD = data["elastic_password"]
  CLOUD_ID = data["elastic_host"]
  # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------------------COMIENZO DEL SCRIPT----------------------------------------------------------------
  # Nombre del Datafeed
  datafeed_id = "datafeed-" + jobname

  # Creacion de log de errores
  current_dateTime = datetime.now().astimezone(timezone('Europe/Madrid'))
  log_name = str(current_dateTime.year) + str("_") + str(current_dateTime.month) + str("_") + str(current_dateTime.day) + str("_") + str(current_dateTime.hour) + str("_") + str(current_dateTime.minute) + str("_") + str(current_dateTime.second) + str("_log.txt")
  log_anomalies_elastic = open(log_name, "w")

  # Calculo del offset por cambio de hora (valido hasta 2027)

  # Timestamps fechas incidencia
  try:
    fecha_inicio_incidencia = datetime.strptime(fecha_inicio_incidencia, '%Y-%m-%d %H:%M:%S')
    fecha_inicio_incidencia_timestamp = datetime.timestamp(fecha_inicio_incidencia)*1000
  except:
    s = "Error: Fecha de inicio de incidencia en formato incorrecto. Error en linea 86.\n"
    log_anomalies_elastic.write(s)
    count_error = count_error + 1
  else:
    try:
      fecha_fin_incidencia = datetime.strptime(fecha_fin_incidencia, '%Y-%m-%d %H:%M:%S')
      fecha_fin_incidencia_timestamp = datetime.timestamp(fecha_fin_incidencia)*1000
    except:
      s = "Error: Fecha de fin de incidencia en formato incorrecto. Error en linea 94.\n"
      log_anomalies_elastic.write(s)
      count_error = count_error + 1
    else:
      if (fecha_inicio_incidencia_timestamp > fecha_fin_incidencia_timestamp):
        s = "Error: Fecha de fin es anterior a la fecha de comienzo. Error en linea 99.\n"
        log_anomalies_elastic.write(s)
        count_error = count_error + 1
      else:
        if ((fecha_fin_incidencia_timestamp - fecha_inicio_incidencia_timestamp) > diferencia_fechas_horas_max*3600*1000):
          s = "Error: La diferencia de fechas es mayor al maximo permitido. Error en linea 104.\n"
          log_anomalies_elastic.write(s)
          count_error = count_error + 1
        else:          
          # Timestamps de las fechas de cambio de hora
          if (datetime.now().year == 2022):
            fecha_cambio_hora_1 = str(datetime.now().year) + "-03-27 02:00:00"
            fecha_cambio_hora_2 = str(datetime.now().year) + "-10-30 02:00:00"
          elif (datetime.now().year == 2023):
            fecha_cambio_hora_1 = str(datetime.now().year) + "-03-26 02:00:00"
            fecha_cambio_hora_2 = str(datetime.now().year) + "-10-29 02:00:00"
          elif (datetime.now().year == 2024):
            fecha_cambio_hora_1 = str(datetime.now().year) + "-03-31 02:00:00"
            fecha_cambio_hora_2 = str(datetime.now().year) + "-10-27 02:00:00"
          elif (datetime.now().year == 2025):
            fecha_cambio_hora_1 = str(datetime.now().year) + "-03-30 02:00:00"
            fecha_cambio_hora_2 = str(datetime.now().year) + "-10-26 02:00:00"
          else:
            fecha_cambio_hora_1 = str(datetime.now().year) + "-03-29 02:00:00"
            fecha_cambio_hora_2 = str(datetime.now().year) + "-10-25 02:00:00"

		  # Primer cambio de hora
          fecha_cambio_hora_1 = datetime.strptime(fecha_cambio_hora_1, '%Y-%m-%d %H:%M:%S')
          fecha_cambio_hora_1_timestamp = datetime.timestamp(fecha_cambio_hora_1)*1000

          # Segundo cambio de hora
          fecha_cambio_hora_2 = datetime.strptime(fecha_cambio_hora_2, '%Y-%m-%d %H:%M:%S')
          fecha_cambio_hora_2_timestamp = datetime.timestamp(fecha_cambio_hora_2)*1000

          # Comparamos fechas en el mismo año 
          fecha_inicio_incidencia_cambio_timestamp = (datetime(datetime.now().year, fecha_inicio_incidencia.month, fecha_inicio_incidencia.day, fecha_inicio_incidencia.hour, fecha_inicio_incidencia.minute, fecha_inicio_incidencia.second).timestamp())*1000 

          if (fecha_inicio_incidencia_cambio_timestamp < fecha_cambio_hora_1_timestamp):
            #offset_1 = -3600
            offset_1 = 0

          if ((fecha_inicio_incidencia_cambio_timestamp > fecha_cambio_hora_1_timestamp) and (fecha_inicio_incidencia_cambio_timestamp < fecha_cambio_hora_2_timestamp)):
            #offset_1 = -7200
            offset_1 = 0

          if (fecha_cambio_hora_2_timestamp < fecha_inicio_incidencia_cambio_timestamp):
            #offset_1 = -3600  
            offset_1 = 0

          fecha_fin_incidencia_cambio_timestamp = (datetime(datetime.now().year, fecha_fin_incidencia.month, fecha_fin_incidencia.day, fecha_fin_incidencia.hour, fecha_fin_incidencia.minute, fecha_fin_incidencia.second).timestamp())*1000 

          if (fecha_fin_incidencia_cambio_timestamp < fecha_cambio_hora_1_timestamp):
            #offset_2 = -3600 
            offset_2 = 0

          if ((fecha_fin_incidencia_cambio_timestamp > fecha_cambio_hora_1_timestamp) and (fecha_fin_incidencia_cambio_timestamp < fecha_cambio_hora_2_timestamp)):
            #offset_2 = -7200
            offset_2 = 0

          if (fecha_cambio_hora_2_timestamp < fecha_fin_incidencia_cambio_timestamp):
            #offset_2 = -3600  
            offset_2 = 0

          # Transformacion de variables de entrada
          # Fechas
          fecha_inicio_incidencia = (fecha_inicio_incidencia_timestamp + offset_1*1000)/1000
          fecha_inicio_incidencia_snapshot = fecha_inicio_incidencia*1000
          fecha_fin_incidencia    = (fecha_fin_incidencia_timestamp + offset_2*1000)/1000
          fecha_fin_incidencia_snapshot = fecha_fin_incidencia*1000

          # Avanzamos esta ultima fecha 2 hora 
          fecha_fin_incidencia_snapshot_avanc = str((fecha_fin_incidencia+7200)*1000)

          fecha_inicio_incidencia_datetime = datetime.fromtimestamp(fecha_inicio_incidencia)
          fecha_fin_incidencia_datetime = datetime.fromtimestamp(fecha_fin_incidencia)

          fecha_inicio_incidencia_datetime_str = str(fecha_inicio_incidencia_datetime)
          fecha_fin_incidencia_datetime_str = str(fecha_fin_incidencia_datetime)
          fecha_inicio_incidencia_datetime_querie = fecha_inicio_incidencia_datetime_str[0:10] + 'T' + fecha_inicio_incidencia_datetime_str[11:]
          fecha_fin_incidencia_datetime_querie    = fecha_fin_incidencia_datetime_str[0:10] + 'T' + fecha_fin_incidencia_datetime_str[11:] 

if count_error == 0:

  # Nombre del Datafeed
  datafeed_id = "datafeed-" + jobname

  try:
  # Conexion con el servidor de Elastic
    es_client = Elasticsearch(
        cloud_id=CLOUD_ID,
        basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
    )
    es_client.info()
  except:
    s = "Error en la conexion con el servidor de Elastic. Comprueba los credenciales o la direccion. Error en linea 169.\n"
    log_anomalies_elastic.write(s)
    print("El script ha finalizado con errores. Ver log de errores")
  else:
    cliente_index = IndicesClient(client=es_client)
    cliente_ml = es_client.ml

    # Creacion del Indice de Backup
    indexname = 'indicebackup'
    indexname_2 = 'indicebackup2'
    indexname_3 = "indicebackup3"

    try:
      cliente_index.create(index=indexname)
    except:
      try:
        cliente_index.delete(index=indexname)
        cliente_index.create(index=indexname)
      except:
        count_error_2 = count_error_2 + 1
    try:
      cliente_index.create(index=indexname_2)
    except:
      try:
        cliente_index.delete(index=indexname_2)
        cliente_index.create(index=indexname_2)
      except:
        count_error_2 = count_error_2 + 1
    try:
      cliente_index.create(index=indexname_3)
    except:
      try:
        cliente_index.delete(index=indexname_3)
        cliente_index.create(index=indexname_3)
      except:
        count_error_2 = count_error_2 + 1
    if (count_error_2 == 0):
      try:
        # Reindexar datos de la anomalia al indice backup 
        es_client.reindex(body={
              "source": {
                  "index": index_data,
                  "query": {
                      "range":{
                          "@timestamp":{
                              "gte": fecha_inicio_incidencia_datetime_querie,
                              "lte": fecha_fin_incidencia_datetime_querie
                              }
                          }
                        }
                      },
                      "dest": {
                          "index": indexname
                      }})
      except:
        s = "Error: No es posible la operacion de reindexado. Error en linea 228.\n"
        log_anomalies_elastic.write(s)
        print("El script ha finalizado con errores. Ver log de errores")
      else:
        try:
          # Obtener datos de la prediccion del Job para el tiempo de anomalia
          df = ed.DataFrame(es_client, es_index_pattern=".ml-anomalies-shared")
        except:
          s = "Error: No es posible obtener datos de prediccion del índice ml-anomalies-shared. Error en linea 235.\n"
          log_anomalies_elastic.write(s)
          print("El script ha finalizado con errores. Ver log de errores")
        else:
          # Filtramos para nuestro job de interes
          df = df[df["job_id"] == jobname]
          # Obtenemos las columnas de interes del dataframe
          df_select = df[["timestamp", "partition_field_value", "typical"]]
          # Eliminar filas con valores nulos
          df_select = ed.eland_to_pandas(df_select)
          df_select = df_select.dropna()
          df_select.head(20)
          len_df = df_select.shape[0]
          array_typical = np.zeros(len_df)
          for i in range(0, len_df):
            array_typical[i] = np.array(df_select["typical"].values[i], dtype=float)
          df_select["typical"] = array_typical
          # Ordenamos filas por timestamp
          df_select.timestamp
          # Seleccionamos timestamp de interes
          df_select['timestamp'] = pd.to_datetime(df_select.timestamp)
          df_select_order = df_select.sort_values(by="timestamp", ascending=True)
          df_select_order = df_select_order.set_index("timestamp")
          df_select_filter = df_select_order.loc[fecha_inicio_incidencia_datetime:fecha_fin_incidencia_datetime]

          # Hacer un rename del nombre de columna "model_median" al nombre de la variable que se predice en el JOB
          df_select_filter.rename(columns = {'typical':variable_name_predict}, inplace = True)

          # Eliminar variable upper_name
          df_select_filter = df_select_filter.reset_index()

          # Crear la pipeline de ingesta para @timestamp 
          client_Ingest = IngestClient(client=es_client)

          try:
            client_Ingest.put_pipeline(id='procesador_date_anomalias', body={
                'description': "Parsea el @timestamp",
                'processors': [
                    {
                        "date": {
                            "field": "timestamp", 
                            "formats": ["ISO8601"],
                            "target_field": "@timestamp"
                            }
                    }
                    ]})
          except:
            s = "Error: No es posible crear una pipeline de ingesta en el servidor. Error en linea 277.\n"
            log_anomalies_elastic.write(s)
            print("El script ha finalizado con errores. Ver log de errores")
          else:
            try:
              # Introducir datos a un indice intermedio
              df_select_filter
              for i, row in df_select_filter.iterrows():
                  doc = {
                      "timestamp": row["timestamp"],
                      variable_name_predict: row[variable_name_predict],
                      partition_field: row["partition_field_value"]
                  }
                  es_client.index(index=indexname_2, id=i, document=doc)
            except:
              s = "Error: No es posible introducir documentos nuevos a un indice. Error en linea 290.\n"
              log_anomalies_elastic.write(s)
              print("El script ha finalizado con errores. Ver log de errores")
            else:
              try:
                # Eliminar del indice principal datos para el timestamp de la incidencia
                es_client.delete_by_query(index=index_data, body={
                  "query": {
                    "range":{
                        "@timestamp":{
                            "gte": fecha_inicio_incidencia_datetime_querie,
                            "lte": fecha_fin_incidencia_datetime_querie
                          }
                        }
                  }
                })
                sleep(5)
              except:
                s = "Error: No es posible eliminar documentos de un indice. Error en linea 307.\n"
                log_anomalies_elastic.write(s) 
                print("El script ha finalizado con errores. Ver log de errores")      
              else:                   
                try:
                  # Parsear a un indice los datos teniendo cuidado con el timestamp al indice principal
                  es_client.reindex(body={
                        'source': {
                            "index": indexname_2
                            },
                            'dest': {
                                "index": index_data,
                                "pipeline": "procesador_date_anomalias"
                                }},
                                wait_for_completion=True)
                  sleep(5)
                except:
                  s = "Error: No es posible reindexar documentos a un indice. Error en linea 323.\n"
                  log_anomalies_elastic.write(s)
                  print("El script ha finalizado con errores. Ver log de errores")        
                else:                      
                  try:
                    # Parsear a un indice los datos teniendo cuidado con el timestamp al indice principal
                    es_client.reindex(body={
                          'source': {
                              "index": indexname_2
                              },
                              'dest': {
                                  "index": index_data,
                                  "pipeline": "procesador_date_anomalias"
                                  }},
                                  wait_for_completion=True)
                  except:
                    s = "Error: No es posible reindexar documentos a un indice. Error en linea 338.\n"
                    log_anomalies_elastic.write(s) 
                    print("El script ha finalizado con errores. Ver log de errores")
                  else:
                    try:
                      # Resetear Job desde la snapshot (buscar snapshot como en el modelo anterior y correr job)
                      # Paso 1. Parar y cerrar el Job
                      cliente_ml.close_job(job_id=jobname)
                    except:
                      s = "Error: No es posible cerrar un job de ML de Elastic. Error en linea 346.\n"
                      log_anomalies_elastic.write(s)   
                      print("El script ha finalizado con errores. Ver log de errores")
                    else:
                      try:
                        # Paso 2. Escoger el id de la snapshot con latest_record_time_stamp mas cercano a la incidencia
                        snapshots_available = cliente_ml.get_model_snapshots(job_id=jobname)
                      except:
                        s = "Error: No es posible obtener el id de un snapshot de un job de ML de Elastic. Error en linea 353.\n"
                        log_anomalies_elastic.write(s) 
                        print("El script ha finalizado con errores. Ver log de errores")
                      else:
                        # Recoger el count 
                        number_snapshots = snapshots_available["count"]
                        if (number_snapshots < 1):
                          s = "Error: No hay snaphsots disponibles. Error en linea 359.\n"
                          log_anomalies_elastic.write(s)     
                          print("El script ha finalizado con errores. Ver log de errores")                                
                        else:
                          # Iterar guardando en una matriz el id del snpashot y latest_record_time_stamp
                          snapshots_info = snapshots_available["model_snapshots"]
                          latest = np.zeros((number_snapshots, 2))
                          for i in range(number_snapshots):
                            latest[i,0] = snapshots_info[i]["latest_record_time_stamp"]

                          for i in range(number_snapshots):
                            latest[i,1] = snapshots_info[i]["snapshot_id"]

                          # Comparar con la fecha de la incidencia para buscar cual esta inmediatamente antes
                            # Ordenar por timestamp (primera columna)
                            latest1 = latest[latest[:,0].argsort()]
                            latest2 = np.flip(latest1, axis=0)
                            # Eliminar valores mayores
                            latest3 = latest2[latest2[:,0]<float(fecha_inicio_incidencia_snapshot)]
                          # Escoger el id de la reversion correcta
                          try:
                            id_revert = latest3[0,1]
                            id_revert = str(int(id_revert))
                          except:
                            s = "Error: No hay disponible snapshot anterior a la fecha de incidencia. Error en linea 380.\n"
                            log_anomalies_elastic.write(s) 
                            print("El script ha finalizado con errores. Ver log de errores")
                          else:
                            try:
                              # Paso 3. Revertir el Job antes de la incidencia 
                              cliente_ml.revert_model_snapshot(job_id=jobname, snapshot_id=id_revert, delete_intervening_results=True)
                            except:
                              s = "Error: No es posible revertir a una snapshot anterior. Error en linea 387.\n"
                              log_anomalies_elastic.write(s)    
                              print("El script ha finalizado con errores. Ver log de errores")  
                            else:
                              try:                                   
                                # Paso 4. Abrir el Job 
                                cliente_ml.open_job(job_id=jobname)
                              except:
                                s = "Error: No es posible abrir un job de ML de Elastic. Error en linea 394.\n"
                                log_anomalies_elastic.write(s) 
                                print("El script ha finalizado con errores. Ver log de errores")
                              else:                                         
                                # Paso 5. Avanzamos datafeed hasta la fecha de inicio de incidencia
                                try:
                                  cliente_ml.start_datafeed(datafeed_id=datafeed_id, end=str(fecha_fin_incidencia_snapshot))
                                except:
                                  s = "Error: No es posible abrir un job ML de Elastic. Error en linea 394.\n"
                                  log_anomalies_elastic.write(s) 
                                  print("El script ha finalizado con errores. Ver log de errores")
                                else:
                                  state_datafeed = 1
                                  # Una vez completado el paso continuamos al siguiente
                                  try:
                                    while (state_datafeed == 1):
                                      sleep(10)
                                      state_datafd = cliente_ml.get_datafeed_stats(datafeed_id=datafeed_id)
                                      st = state_datafd['datafeeds']
                                      state_actual = st[0]['state'] 
                                      if (state_actual == 'stopped'):
                                        state_datafeed = 0
                                    print("Eliminacion de la incidencia con éxito")
                                  except:
                                    s = "Error: No es posible eliminar la incidencia con éxito. Error en el bucle de la línea 407.\n"
                                    log_anomalies_elastic.write(s)   
                                    print("El script ha finalizado con errores. Ver log de errores")
                                  else:
                                    # Sustituir datos del indice backup al indice original para el tiempo de incidencia
                                    # 1. Eliminar datos ingestados de la prediccion para el tiempo de incidencia
                                    try:
                                      es_client.delete_by_query(index=index_data, body={
                                        "query": {
                                          "range":{
                                              "@timestamp":{
                                                  "gte": fecha_inicio_incidencia_datetime_querie,
                                                  "lte": fecha_fin_incidencia_datetime_querie
                                                }
                                              }
                                        }
                                      })
                                      sleep(5)
                                    except:
                                      s = "Error: No es posible eliminar datos de indice. Error en línea 435.\n"
                                      log_anomalies_elastic.write(s) 
                                      print("El script ha finalizado con errores. Ver log de errores")
                                    else:   
                                      try:                                                                                          
                                        # 2. Ingesta de los datos del backup para el tiempo de incidencia
                                        es_client.reindex(body={
                                              "source": {
                                                  "index": indexname,
                                                  "query": {
                                                      "range":{
                                                          "@timestamp":{
                                                              "gte": fecha_inicio_incidencia_datetime_querie,
                                                              "lte": fecha_fin_incidencia_datetime_querie
                                                              }
                                                          }
                                                        }
                                                      },
                                                      "dest": {
                                                          "index": index_data
                                                      }},
                                                      wait_for_completion=True)
                                        sleep(5)
                                      except:
                                        s = "Error: No es posible ingestar datos en indice. Error en línea 441.\n"
                                        log_anomalies_elastic.write(s) 
                                        print("El script ha finalizado con errores. Ver log de errores")
                                      else:
                                        # Eliminar indices creados
                                        try:
                                          cliente_index.delete(index=indexname)
                                        except:
                                          s = "Error: No es posible eliminar indices. El script continuará sin borrarlos. Error en línea 462.\n"
                                          log_anomalies_elastic.write(s)
                                        try:
                                          cliente_index.delete(index=indexname_2)
                                        except:
                                          s = "Error: No es posible eliminar indices. El script continuará sin borrarlos. Error en línea 468.\n"
                                          log_anomalies_elastic.write(s)
                                        try:
                                          cliente_index.delete(index=indexname_3)
                                        except:
                                          s = "Error: No es posible eliminar indices. El script continuará sin borrarlos. Error en línea 474.\n"
                                          log_anomalies_elastic.write(s)
                                        # Paso 6. Avanzamos datafeed hasta el tiempo real
                                        try:
                                          cliente_ml.open_job(job_id=jobname)
                                          sleep(5)
                                        except:
                                          s = "Error: No es posible abrir el job de de ML de Elastic. Error en línea 501.\n"
                                          log_anomalies_elastic.write(s) 
                                          print("El script ha finalizado con errores. Ver log de errores")
                                        else:
                                          try:
                                            cliente_ml.start_datafeed(datafeed_id=datafeed_id, start=str(fecha_fin_incidencia_snapshot))
                                          except:
                                            s = "Error: No es posible comenzar el datafeed del job de de ML de Elastic. Error en línea 508.\n"
                                            log_anomalies_elastic.write(s) 
                                            print("El script ha finalizado con errores. Ver log de errores")                                                      
                                            log_anomalies_elastic.close()
                                          else:
                                            s = "El script ha finalizado con exito.\n"
                                            log_anomalies_elastic.write(s) 
                                            print("El script ha finalizado con exito.")    
    else:
      s = "Error en la creacion de indices en Elastic. Error en linea 189.\n"
      log_anomalies_elastic.write(s)
      print("El script ha finalizado con errores. Ver log de errores")                                                                                                    
else:
  print("El script ha finalizado con errores. Ver log de errores")
  log_anomalies_elastic.close()