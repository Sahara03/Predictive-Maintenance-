# Predictive-Maintenance
2 Predictive Models created with PySpark, Hadoop and Docker

# Requeriments:
-Ubuntu /
-Docker 

# Initial Steps:
  For Hadoop:
  sudo docker pull sahara03/hadoop
  sudo docker run -it --name hadoop sahara03/hadoop

  For Models:
  sudo docker pull jupyter/pyspark-notebook
  sudo docker run -it --name GBC -p 8888:8888 jupyter/pyspark-notebook
  sudo docker run -it --name LSTM -p 8888:8888 jupyter/pyspark-notebook

  For Postgres:
  sudo docker pull sahara03/postgres
  sudo docker run -it --name posgres sahara03/postgres
  
 # Create Network
 sudo docker network create poc
 sudo docker connect poc hadoop
 sudo docker connect poc GBC (or LSTM)
 sudo docker connect poc postgres
 
 # Dockers up
 sudo docker start -i Hadoop
 sudo docker start -i GBC (or LSTM)
 sudo docker start -i postgres
 
 For GBC copy the files (GBC Files) on jupyter
 For LSTM copy the files (LSTM files) on jupyter
 
## Done!
 
### The images of dockers you can found in this link too
https://hub.docker.com/u/sahara03
