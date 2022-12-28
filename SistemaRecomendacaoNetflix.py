#!pip install -U scikit-learn
#!pip install pyspark
#!pip install findspark

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col,array_contains,explode

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

diretorio = "dados/training_set/"

#Importando os arquivos para o Spark
MAX_MEMORY = "15g"

spark = SparkSession.builder.appName('loadNetflix') \
            .config('spark.master', 'local')\
            .config("spark.executor.memory", MAX_MEMORY) \
            .config("spark.driver.memory", MAX_MEMORY) \
            .getOrCreate()

schemaMovies = StructType() \
      .add("Movie_Id",IntegerType(), True) \
      .add("Year",StringType(), True) \
      .add("MovieName",StringType(), True)

schema = StructType() \
      .add("Movie_Id",IntegerType(), True) \
      .add("Cust_Id",IntegerType(), True) \
      .add("Rating",IntegerType(), True) \
      .add("Date",StringType(), True) \


spark.conf.set('spark.sql.pivotMaxValues', u'50000')

df = spark.read.options(delimiter=',').option("header", False).schema(schema).csv(diretorio + "tratado/") 
dfMovies = spark.read.options(delimiter=';').option("header", False).schema(schemaMovies).csv("dados/movie_titles_f.txt")

df = df.join(dfMovies, ['Movie_Id'], 'left')

(training, test) = df.randomSplit([0.8, 0.2]) #divide o df em porções para treinamento e teste

#Adicionando A Ero do Gelo e Rei Leao e Tom e Jerry
columns = ['Movie_Id', 'Cust_Id', 'Rating', 'Date']
vals = [ (8743, 9999991, 5, '2022-10-16'), \
         (16660, 9999991, 5, '2022-10-16'), \
         (976, 9999991, 5, '2022-10-16'), \
            
        (6001, 9999991, 5, '2022-10-16'), \
        (3079, 9999991, 5, '2022-10-16'), \
        (16222, 9999991, 5, '2022-10-16'), \
        ]
dfNewCust = spark.createDataFrame(vals, columns)
dfNewCust = dfNewCust.join(dfMovies, ['Movie_Id'], 'left')

training = training.union(dfNewCust)

als = ALS(maxIter=10, regParam=0.01, userCol="Cust_Id", itemCol="Movie_Id", ratingCol="Rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


topredict = training[ training['Cust_Id'] == 9999991 ] 
predictions = model.transform(topredict)


predictions = model.recommendForUserSubset(topredict.select('Cust_Id').distinct(), 20) \
                    .withColumn("rec_exp", explode("recommendations")) \
                    .select('Cust_Id', col("rec_exp.Movie_Id"), col("rec_exp.rating"))

dfPredictions = predictions.join(dfMovies, ['Movie_Id'], 'left') \
                         .join(topredict, ['Movie_Id'], 'left_anti') \
                         .filter( predictions.rating > 0  ) \
                         .sort( col('rating').desc() )

#Imprime o DataFrame
print( dfPredictions.toPandas() )

#Tratamento para colocar na primeira coluna o Codigo do Filme ao inves da primeira linha
def trataArquivoPontuacao(numero):    
    arquivoOrigem = diretorio + "mv_" + str(numero).zfill(7) + ".txt"
    arquivoDestino = diretorio + "tratado/mv_" + str(numero).zfill(7) + ".txt"
    
    arquivo1 = open( arquivoOrigem, 'r') # Abra o arquivo (leitura)
    arquivo2 = open( arquivoDestino, 'w') # Abre novamente o arquivo (escrita)

    pulouLinha1 = False
    for item in arquivo1:
        if pulouLinha1:
            arquivo2.write(str(numero)+','+ item )
        else:
            pulouLinha1 = True

    arquivo1.close()
    arquivo2.close()

fim =  17770
for i in range(1, fim+1):
    trataArquivoPontuacao(i)


def trataArquivoFilmes():
    arquivo1 = open( 'dados/movie_titles.txt', 'r') # Abra o arquivo (leitura)
    arquivo2 = open( 'dados/movie_titles_f.txt', 'w') # Abre novamente o arquivo (escrita)

    for item in arquivo1:
    
        linha = list(item)
        if linha[1] == ",":
            linha[1] = ';'
            linha[6] = ';'
        else:
            if linha[2] == ',':
                linha[2] = ';'
                linha[7] = ';'
            else:
                if linha[3] == ',':
                    linha[3] = ';'
                    linha[8] = ';'
                else:
                    if linha[4] == ',':
                        linha[4] = ';'
                        linha[9] = ';'
                    else:
                        if linha[5] == ',':
                            linha[5] = ';'
                            linha[10] = ';'

        arquivo2.write(''.join(linha))    # escreva o conteúdo criado anteriormente nele.

    arquivo1.close()
    arquivo2.close()

trataArquivoFilmes()