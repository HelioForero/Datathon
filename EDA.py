# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder


df = pd.read_csv("./hospitalizaciones_train.csv")

df.head()

# %%
display(df.info())

df.nunique()


# %% [markdown]
# Primero generamos la salida a partir de la columna 'Stay (in days)'

# %%
df['output'] = df['Stay (in days)'].map(lambda x: 1 if x > 8 else 0)
#df.drop(['Stay (in days)'], axis=1,inplace=True)



# %%
df.describe()

# %% [markdown]
# Ahora sabemos que la probabilidad de estar mas 8 dias interno es de 62% en general. Ahora revisamos las columnas con variables categoricas para definir como hacer el encode

# %%
df_genders = df.groupby('gender').count()
df_genders.plot.pie(y='output')



df_genders = df.groupby('gender').mean()
df_genders['output'] = df_genders['output'] *100
display(df_genders.head(4))
axis = df_genders.plot.bar(y='output', rot=0, title= 'Probability to stay over 8 days by gender') #color={"Male": "red", "Female": "blue", "Other":"green"})

# Teniendo en cuenta que si afecta el genero en la probabilidad, vamos a generar un encoder one hot, con 'Female' como 0 y "Male" y "Other" como 1 
# pues tienen el mismo comportamiento en la variable objetivo

# %%
df_conditions = df.groupby('health_conditions').count()
df_conditions.plot.pie(y='output')

df_conditions = df.groupby('health_conditions').mean()
df_conditions['output'] = df_conditions['output']*100
display(df_conditions.head(6))
df_conditions.plot.bar(y='output', title='Probability to stay over 8 days by Health Conditions')

# Vemos que las condiciones de salud no tienen un impacto fuerte en en la variable objetivo

# %%
df_department = df.groupby('Department').count()
df_department.plot.pie(y='output')
display(df_department.head(5))

df_department = df.groupby('Department').mean()
df_department['output'] = df_department['output']*100
display(df_department.head(5))
df_department.plot.bar(y='output', title='Probability to stay over 8 days by Department')

# Observamos que si hay un impacto en la variable objetivo dependiendo del departamento. En este caso se usará un label encoder 
# y los valores se asignaran acorde a la probabilidad. 

# %%
df_ward = df.groupby('Ward_Facility_Code').count()
df_ward.plot.pie(y='output')

df_ward = df.groupby('Ward_Facility_Code').mean()
df_ward['output'] = df_ward['output']*100
display(df_ward.head(6))
df_ward.plot.bar(y='output', title='Probability to stay over 8 days by Department')

# En este caso se observa que si hay un ligero impacto del departamento de admision. En este caso podemos usar un label encoder, ordenando por el promedio

# %%
df_age = df.groupby('Age').count()
df_age.plot.pie(y='output')


df_age = df.groupby('Age').mean()
df_age['output'] = df_age['output']*100

df_age.plot.bar(y='output', title='Probability to stay over 8 days by Age Group')

df_age.head(20)
# En este caso se distingue una fuerte relacion en los rangos de edad y la variable objetivo. En este caso podemos re-categorizar a 3 variables,
#  menores de 10 años, entre 11 y 50, y mayores de 50

# %%
df_admision = df.groupby('Type of Admission').count()
df_admision.plot.pie(y='output')


df_admision = df.groupby('Type of Admission').mean()
df_admision['output'] = df_admision['output']*100

df_admision.plot.bar(y='output', title='Probability to stay over 8 days by Admision type')

# No hay una relacion fuerte entre la variable de salida y el tipo de admision. Podemos considerar descartar esta variable

# %%
df_severity = df.groupby('Severity of Illness').count()
df_severity.plot.pie(y='output')

df_severity = df.groupby('Severity of Illness').mean()
df_severity['output'] = df_severity['output']*100

df_severity.plot.bar(y='output', title='Probability to stay over 8 days by Severity of Ilness')

# Vemos que esta clasificacion tiene muy poco impacto en la variable de salida. Podemos considerar descartar esta variable

# %%
df_insurance = df.groupby('Insurance').count()
df_insurance.plot.pie(y='output')


df_insurance = df.groupby('Insurance').mean()
df_insurance['output'] = df_insurance['output']*100

df_insurance.plot.bar(y='output', title='Probability to stay over 8 days depending on Insurance Status')

# El impacto de esta variable es despreciable en este analisis

# %%
df_combo = df.groupby(['Department','Ward_Facility_Code']).mean()
df_combo['output'] = df_combo['output']*100

df_combo.plot.bar(y='output') #  title='Probability to stay over 8 days by Department and Facility',

df_combo.head(20)
#considerando que el departamento esta ligeramente asociado a la seccion del hospital, revisamos agrupando ambos

# observamos que si bien hay una relación fuerte entre ambas cosas y podriamos considerdad usar solo una de las 
# dos al tener la misma información, pero la disparidad en radioterapia hace que considere retenr ambas variables

# %%
df_combo = df.groupby(['Department','gender']).mean()
df_combo['output'] = df_combo['output']*100

df_combo.plot.bar(y='output') #  title='Probability to stay over 8 days by Department and Facility',

df_combo.head(20)
# en este caso vemos que la informacion que tenemos de si es admitido en ginecologia y si es mujer, puede ser redundante. 
# Se puede considerar no tener en cuenta uno de los dos, en este caso tendria mas sentido no usar genero

# %%
df_combo = df.groupby(['Department','doctor_name']).mean()
df_combo['output'] = df_combo['output']*100

df_combo.plot.bar(y='output') #  title='Probability to stay over 8 days by Department and Facility',

# de aqui podemos concluir que la informacion que nos da el nombre del doctor, ya la tenemos en el departemento de admision

df_combo.head(20)

# %%
df_dr = df.groupby('doctor_name').count()
df_dr.plot.pie(y='output')

df_dr = df.groupby('doctor_name').mean()

df_dr['output'] = df_dr['output']*100

df_dr.plot.bar(y='output', title='Probability to stay over 8 days by Atending Dr')

# Se puede observar que el doctor que atiende tiene una influencia en la variable objetivo. 
# En este caso podemos agrupar los doctores en 3 grupos dependiendo de su influencia. 

# %%
dfg = df[df['Department'] == 'gynecology']

df_age = dfg.groupby('Age').mean()
df_age['output'] = df_age['output']*100

df_age.plot.bar(y='output', title='Probability to stay over 8 days by Age Group')

df_age.head(20)


# %%
df_dr = dfg.groupby('doctor_name').count()
df_dr.plot.pie(y='output')

df_dr = dfg.groupby('doctor_name').mean()

df_dr['output'] = df_dr['output']*100
display(df_dr.head(6))
df_dr.plot.bar(y='output', title='Probability to stay over 8 days by Atending Dr')

# %%
df_conditions = dfg.groupby('health_conditions').count()
df_conditions.plot.pie(y='output')

df_conditions = dfg.groupby('health_conditions').mean()
df_conditions['output'] = df_conditions['output']*100
display(df_conditions.head(6))
df_conditions.plot.bar(y='output', title='Probability to stay over 8 days by Health Conditions')

# %%
df_ward = dfg.groupby('Ward_Facility_Code').count()
df_ward.plot.pie(y='output')

df_ward = dfg.groupby('Ward_Facility_Code').mean()
df_ward['output'] = df_ward['output']*100
display(df_ward.head(6))
df_ward.plot.bar(y='output', title='Probability to stay over 8 days by Department')

# %%
df_admision = dfg.groupby('Type of Admission').count()
df_admision.plot.pie(y='output')


df_admision = dfg.groupby('Type of Admission').mean()
df_admision['output'] = df_admision['output']*100
display(df_admision.head(6))
df_admision.plot.bar(y='output', title='Probability to stay over 8 days by Admision type')

# %%
df_severity = dfg.groupby('Severity of Illness').count()
df_severity.plot.pie(y='output')

df_severity = dfg.groupby('Severity of Illness').mean()
df_severity['output'] = df_severity['output']*100
display(df_severity.head(6))
df_severity.plot.bar(y='output', title='Probability to stay over 8 days by Severity of Ilness')


# %%
df_insurance = dfg.groupby('Insurance').count()
df_insurance.plot.pie(y='output')


df_insurance = dfg.groupby('Insurance').mean()
df_insurance['output'] = df_insurance['output']*100
display(df_insurance.head(6))
df_insurance.plot.bar(y='output', title='Probability to stay over 8 days depending on Insurance Status')

# %%
import seaborn as sns
corr = dfg.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True)

# %% [markdown]
# En resumen, las variables que tienen un impacto fuerte en la variable objetivo son: 
# + Genero
# + Departamento
# + Sección
# + Rango Etario
# + Dr que atiende 
# 
# Adicionalemente, la sección solo influyé fuertemente para el departamento de radiotherapy 

# %% [markdown]
# Iniciamos los cambios en el Dataframe, primero codificando Genero



# %%
df_train = df[['Department','gender','Ward_Facility_Code', 'Age', 'Available Extra Rooms in Hospital', 
               'staff_available', 'Admission_Deposit','Severity of Illness' , 'doctor_name',
               'Stay (in days)', 'output']] # 'gender',


# Como vimos anteriormente, el genero que tiene impacto en la variable objetivo es 'Female'. Los otros dos no tienen distincion en su impacto entre ellos. 
# en este caso, solo tomo 'Female' como 1 y cualquier otro como 0

#df_train['gender'] = df_train.gender.map(lambda x: 1 if x == 'Female' else 0)

# def uncode_gender(gender): # Creamos funcion para decodificar ya que la hicimos a mano
   
#     if gender == 1:
#         return 'Female'
#     else:
#         return 'Male or Other'


df_train.head(10)


# %% [markdown]
# Ahora procedemos a cambiar Department y Ward_Facility_code combiandolos

# %%

#Observamos que hay seis categorias grandes dentro de estas dos features:

# - Si esta en Cirugia o Anestesiologia ->5
# - Si esta en TB & Chest disease -> 4
# - si en radio terapia y edificio A -> 3
# - si esta en terapia y edificio C -> 2
# - si esta en terapia y edificio E -> 1
# - si esta en gynecology -> 0


# Creamos codificamos y creamos funcion para decodificar ya que la hicimos a mano

# dw = []

# for ind in df_train.index:
#     if ((df_train.loc[ind, 'Department'] == 'surgery') or (df_train.loc[ind, 'Department'] == 'anesthesia')):
#         dw.insert(ind,5)
#     elif df_train.loc[ind, 'Department'] == 'TB & Chest disease':
#         dw.insert(ind,4)
#     elif ((df_train.loc[ind, 'Department'] == 'radiotherapy') & (df_train.loc[ind, 'Ward_Facility_Code'] == 'A')):
#         dw.insert(ind,3)
#     elif ((df_train.loc[ind, 'Department'] == 'radiotherapy') & (df_train.loc[ind, 'Ward_Facility_Code'] == 'C')):
#         dw.insert(ind,2)
#     elif ((df_train.loc[ind, 'Department'] == 'radiotherapy') & (df_train.loc[ind, 'Ward_Facility_Code'] == 'E')):
#         dw.insert(ind,1) 
#     elif df_train.loc[ind, 'Department'] == 'gynecology':
#         dw.insert(ind,0)
#     else:
#         print(df_train.loc[ind, 'Department'], df_train.loc[ind, 'Ward_Facility_Code'])

# def uncode_depward(depward):
#     if depward == 0:
#         return 'gynecology'
#     elif depward == 1:
#         return 'radiotherapy ward E'
#     elif depward == 2:
#         return 'radiotherapy ward C'
#     elif depward == 3:
#         return 'radiotherapy ward A'
#     elif depward == 4:
#         return 'TB & Chest disease'
#     elif depward == 4:
#         return 'surgery or anesthesia'


# df_train['depward'] = dw

# df_train.drop(['Department', 'Ward_Facility_Code'], axis = 1, inplace=True)

from sklearn import preprocessing
le = preprocessing.OrdinalEncoder()
# df_train['Department'] = le.fit_transform([df_train['Department']])
# df_train['Ward_Facility_Code'] = le.fit_transform([df_train['Ward_Facility_Code']])
# df_train['Severity of Illness'] = le.fit_transform([df_train['Severity of Illness']])
# df_train['doctor_name'] = le.fit_transform([df_train['doctor_name']])
# df_train['Age'] = le.fit_transform([df_train['Age']])
df_train[['gender','Department','Ward_Facility_Code', 
'Severity of Illness', 'doctor_name','Age']] = le.fit_transform(df_train[['gender','Department','Ward_Facility_Code',
 'Severity of Illness', 'doctor_name','Age']])

df_train.head(10)

# %% [markdown]
# Ahora vamos a codificar la edad en 3 categorias:
#  + Menores de 10 y tercera edad (2)
#  + entre 11 y 30 (0)
#  +  Entre 30 y 50 (1)

# %%
# a = []

# for ind in df_train.index:
#     if df_train.loc[ind,'Age'] in ['0-10', '51-60', '61-70','71-80', '81-90', '91-100']:
#         a.insert(ind,2)
#     elif df_train.loc[ind,'Age'] in ['11-20', '21-30']:
#         a.insert(ind,1)
#     elif df_train.loc[ind,'Age'] in ['31-40','41-50']:
#         a.insert(ind,0)
#     else:
#         print(df_train.loc[ind,'Age'])

# def uncode_age(age):
#     if age == 0:
#         return '31-50'
#     if age == 1:
#          return '11-30'
#     if age == 1:
#         return '0-10 or 50+'

# df_train['agecode'] = a

# df_train.drop('Age', inplace=True, axis=1)

# df_train = df_train[['depward', 'agecode', 'Available Extra Rooms in Hospital', 'staff_available',
#  'Admission_Deposit', 'Stay (in days)', 'output']]



df_train.head()


# %%
import seaborn as sns
corr = df_train[df_train['Department']==2].corr()
sns.heatmap(corr, 
        annot=True, 
        cmap= 'coolwarm',  fmt= '.2f')



# %%
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

#sns.pairplot(df_train, diag_kind='kde', hue= 'output')
#plt.show()



# %%
df_train.drop(['staff_available'], axis=1, inplace=True)

df_train.to_csv('./transformedH.csv')


