################################################ EDA: ANÁLISIS EXPLORATORIO DE DATOS

# Preprocesamiento
import pandas as pd
import numpy as np
import unidecode

# Visualización 
import matplotlib.pyplot as plt
import seaborn as sns


# 0. COMPOSICION DEL DATAFRAME

def initial_analysis(df): 
    """Esta función realiza el reconocimiento inicial del dataframe.
         Args: 
            dataframe: df
         Returs:
            Una vision de los datos que contiene el dataframe.
    """
    print(f"El DataFrame contiene {df.shape[0]} filas por {df.shape[1]} columnas.")
    
    print(f"\nSus columnas tiene el nombre de:\n {df.columns}.")
      
    return df.info()





# 1. ESTANDARIZACIÓN DEL NOMBRE DE LAS COLUMNAS DEL DATAFRAME.

# De esta forma se evita la pérdida de tiempo buscando el nombre original de la columna, si tiene espacio - mayúsculas, etc.

def remove_accents(a): 
    """Esta función reemplaza el nombre de las columnas del dataframe si tienen grado, apóstrofe o un punto por un guión bajo.
         Args: 
            columns: df.columns
         Returs:
            Una transformación que estandariza el nombre de las columnas del dataframe.  
    """
    a = a.replace('°',  '')
    a = a.replace("'",  '')
    a = a.replace(".",  '_')
    return unidecode.unidecode(a)  

def process_cols(columns):
    """Esta función transforma el nombre de las columnas del dataframe sin mayúsculas, reemplaza los espacios blanco con guion bajo y el guion medio con guion bajo.
         Args: 
            columns: df.columns
         Returs:
            Una transformación que estandariza el nombre de las columnas del dataframe.  
    """
    columns = columns.str.lower()   
    columns = columns.str.strip()   
    columns = columns.str.strip('.')   
    columns = columns.str.replace(' ', '_')
    columns = columns.str.replace('-', '_') 
    columns = [remove_accents(x) for x in columns]
    return columns




# 2. VALORES PERDIDOS

def datos_faltantes(df):
    """ Esta función muestra la cantidad de datos faltantes en el dataframe.
         Args: 
             data: dataframe con el que se está trabajando. (df)
         Returs:
             El total de valores pedidos por columna.
             El porcentaje de valores perdidos.
             El tipo de datos.
    """ 
    total = df.isnull().sum()
    porcentaje = round((total/df.shape[0])*100,2)
    tt = pd.concat([total, porcentaje], axis=1, keys=['Total', 'Porcentaje'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Tipo de Dato'] = types
    tt = tt[tt['Total'] != 0]
    tt =tt.sort_values('Total',ascending=False) 
    display(tt)
 
    print(color.azul +                                                     "MATRIZ DE VALORES PERDIDOS"+color.fin)
    plt.figure(figsize = (4,4))
    msngo.matrix(df.replace([8, 9], [np.nan, np.nan]))
    plt.show()
    print(color.azul +                                                     "CANTIDAD DE DATOS PERDIDOS POR VARIABLE"+color.fin)
    msngo.bar(df, color = "dodgerblue", sort = "ascending", fontsize = 12)
    plt.show()
    print(color.azul +                                                     "GRADO DE ASOCIACION ENTRE DATOS PERDIDOS POR VARIABLES"+color.fin)
    msngo.heatmap(df)
    plt.show()
    
    df_1 = df.dropna()
    perdida_de_muestra = ((df.shape[0] - df_1.shape[0])/df.shape[0])* 100
    print(f"\n\nSí eliminamos los datos faltantes perdemos el {round(perdida_de_muestra,2)}% de la muestra") 
    
    
    
    
# 3. RECODIFICACIÓN DE VARIABLES

## Recodificación de una variable
def recodificacion(df, vble, renombrar_por, reemplazar, reemplazar_por):
    """ Esta función recodifica una variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vble: la variable que se quiere recodificar.(string)
             renombrar_por: la variable (string)
             reemplazar: los valores que se quieren modificar (list)
             reemplazar_por: valor/nombre que se quiere considerar (list)
         Returs:
             Nuevo df con las modificaciones buscadas.
    """
    print(f"Los valores que asume la variable {vble}/{renombrar_por} originalmente son: \n")
    print(df[vble].value_counts())
    print("                                 ") 
    df = df.rename(columns={vble:renombrar_por})
    df[renombrar_por] = df[renombrar_por].replace(reemplazar, reemplazar_por)
    print(f"La variable transformada {vble}/{renombrar_por} asume los nuevos valores de: \n")
    print(df[renombrar_por].value_counts())
    print("----------------------------------------------------------------------------------- ")  
    return df 
   
## Recodificación binaria de una variable    
def recodificacion_binaria(df, vbles):
    """ Esta función recodifica una variable binaria asignandole el criterio de 1 a aquellas categorías minoritarias.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vbles: las variables que se quiere recodificar.(string)
         Returs:
             Nuevo df con variables recodificada de forma binaria.
    """ 
    for i in vbles:
        print(f"Los valores que asume la variable {i} originalmente son: \n")
        print(df[i].value_counts())
        print("                                 ") 
        df[i] = np.where(df[i] == df[i].value_counts().index[0], 0, 1)
        print(f"La variable transformada {i} asume los nuevos valores de: \n")
        print(df[i].value_counts())
        print("----------------------------------------------------------------------------------- ")
    return df 

# Transformación numérica
def transformacion_numerica(df, vble):
    """ Esta función transforma a números una columna numérica que esta considerada bajo una estructura de datos genérica.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vble: es la variable que esta como string que queremos transformar a número
         Returs:
             La variable transformada de object a float
            
    """    
    df[vble] = df[vble].str.replace('"','')
    df[vble] = pd.to_numeric(df[vble])
    return df 



################################################ MEDIDAS DESCRIPTIVAS DE LOS DATOS


# Diferenciar entre variables cualitativas y cuantitativas
def medidas_descriptivas(df):
    """ Esta función presenta las medidas descriptivas de cada variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
         Returs:
             Medidas descriptivas
    """ 
    print(color.azul + f"                                 VARIABLES CUANTITATIVAS \n                        "+ color.fin)
    display(round(df.describe().transpose(),2))
    print(color.azul + f"                                 VARIABLES CUALITATIVAS \n                        "+ color.fin)
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"{col.upper()}")
            print("----------")
            print(round(df[col].value_counts(),2),"\n") 
            print("----------")
    
            
           
        
################################################ GRAFICOS DE LAS VARIABLES

def grafico_vbles(df):
    """ Esta función gráfica las variables según sean cualitativas o cuantitativas.
         Args: 
             df: dataframe con el que se está trabajando. (df)
         Returs:
             Gráficos con los comportamientos de las variables.
    """ 
    for n, i in enumerate(df):
        j = n + 1
        plt.subplots_adjust(left = 3,right = 7,bottom = 7, top = 25,wspace = 0.2, hspace = 0.2)
        plt.subplot(17, 2 , j)
        #plt.figure(figsize=(5,3))
        
        # Graficos de barras para las categórica
        if type(df[i][2]) == str:
            sns.countplot(y = df[i].dropna(), palette="pastel")
            plt.title(i.upper(), fontsize= 25)
            plt.xlabel("")
            
        else:
        # Histograma para las cuantitativas
            if len(df[i].value_counts()) > 2:
                sns.distplot(df[i].dropna())
                plt.title(i.upper(), fontsize= 25)
                plt.xlabel("")
            else:
                sns.countplot(y = df[i].dropna(), palette="Set3")
                plt.title(i.upper(), fontsize= 25)
                plt.xlabel("")

################################################ OUTLIERS            
            
def outliers(lista,columna, name, df):
    df_train_filtrado = df[[columna, name]]
    outliers = []
    for i in lista:
        df_filtro = df_train_filtrado[df_train_filtrado[columna]== i]
        q1, q3 =df_filtro[name].quantile(0.25), df_filtro[name].quantile(0.75)
        rango_interc = q3 - q1
        upper = q3 + 1.5 * rango_interc
        lower = q1 - 1.5 * rango_interc 
        
        median = df_filtro[name].quantile(0.50)
        print(f"La mediana de {i} es {median}")
        
        down = df_filtro[(df_filtro[name] < lower)]
        up = df_filtro[(df_filtro[name] > upper)]
        if down.empty == False:
            outliers.append(down)
        if up.empty == False:
            outliers.append(up)
       
    return outliers
                
                
                
class color:
    violeta= '\033[95m'
    celeste = '\033[96m'
    azul = '\033[94m'
    verde = '\033[92m'
    amarillo = '\033[93m'
    rojo = '\033[91m'
    negrita = '\033[1m'
    underline = '\033[4m'
    fin = '\033[0m'