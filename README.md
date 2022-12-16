# <h1 align="center">**`Proyecto Individual 2`**
# <h1 align=center>**`Data Engineering`**</h1>

<div style="text-align: right"> Helio Angel Forero Mora</div>
<hr>

En este Proyecto usamos un dataset de pacientes de hospital, en el que se nos dan 14 columnas que describen diferentes clasificaciones relevantes del paciente y en otra columna su estadia en dias

Se nos solicita a partir de esta información, entrenar un modelo que prediga si un paciente sera hospitalizado por mas de 8 días, pues se considera una extadia larga, para acomodar de mejor forma las habitaciones disponibles. Para mayor información de la premisa, revisar el archivo Assigment.md

En este proceso se hace una exploracion de los datos, detallada en EDA.ipynb, donde se escogen las features que mantendremos y se codifican las variables categorias de forma ordinal. Estas se guardan en el archivo transformedH.csv para su uso con diferentes modelos

Los distintos modelos probados, estan en models.ipynb. Por comodidad del lector se dejo de primeras el algoritmo que se uso para efectuar las predicciones. El algoritmo que mejor rendimiento consiguio fue el XGBoost y este algoritmo entranado se guardo, por medio de la libreria pickle, en el archivo xgb_trained.sav. Posteriormente se realizaron las predecciones sobre el archivo hospitalizaciones.test. Estas predecciones se hicieron en el archivo transform.ipynb

