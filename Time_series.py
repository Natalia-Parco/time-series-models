################################################ TIME SERIES

# Preprocesamiento
import pandas as pd
import numpy as np
import unidecode

# Visualización 
import matplotlib.pyplot as plt
import seaborn as sns



# Modelación
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

# Métricas de evaluación
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import kpss
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, r2_score
 

############################### A. STATISTICS
############ UNIVARIATE PREDICTION

# A.0. MOVING AVERAGE
## Este modelo utiliza la media de los valores pasados para predecir los futuros.

    # 0.1 Simple moving average (SMA)

    # 0.2 Exponential moving average(EMA)

    
    
    
# A.1. EXPONENTIAL SMOOTHING
## Este modelo también se basa en la media de los valores pasados, pero da más importancia a los valores más recientes mediante la asignación de pesos exponenciales a los datos históricos.




# A.2. LINEAR REGRESSION
## Este modelo utiliza una relación lineal entre la variable dependiente y una o más variables independientes para predecir los valores futuros

def relacion_lineal(df, y_pred):
    """ Esta función presenta la relación existente entre 'y' predicha y las variables explicativas. Como estamos en un modelo de regresión lineal, la relación existente debe serlo también. De lo contrario se debe transformar la variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir
         Returs:
            Un gráfico de regresión lineal
    """ 
    for i in df.columns:
        sns.regplot(f"{i}", y_pred , data = df, color='darkviolet',marker="+")
    return plt.show()



def regresion_lineal(df, y_pred, X_drop, n, p_value, cte=False, correccion = False, plot = False):
    
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir.
             X_drop: lista con y_pred y las variables que no son significativas.
             n: 0 = Indicadores OLS 
                1 = Indicadores de las variables
                2 = todos los indicadores
             cte = ordenada al origen
             correccion = variables no significativas.
             
         Returs:
              Regresion Lineal
    """ 
    y = df[y_pred]
    X = df.drop(X_drop, axis = 1)
    
    if cte == True:
        X = sm.add_constant(X)

    model = sm.OLS(y,X)
    model = model.fit()
    results= model.summary()
    tabla = model.summary2().tables[1]
    coeficientes = pd.DataFrame(tabla[['Coef.']]).transpose()
    filtrar = tabla[tabla['P>|t|'] > p_value].sort_values('P>|t|',ascending =False)
    lista = list(filtrar.index)   
    
    if n == 2:
        display(results)
    else:
        display(results.tables[n])
    
    if correccion == True:
        print(color.violeta + "                  VARIABLES NO SIGNIFICATIVAS" + color.fin)
        display(filtrar)
        print(color.violeta + f"\nVariables a excluir:  {lista}\n\n" + color.fin)   
        
    #Limpieza de variables no significativas
        lista_resultados = []
        for i in lista:
            drop = X_drop
            drop.append(i)
            y = df[y_pred]
            X = df.drop(drop, axis = 1)
            if cte == True:
                X = sm.add_constant(X)

            model_1= sm.OLS(y,X).fit()
            print(f"\n\nSí eliminamos la variable {i.upper()} nos quedan como:\n")
            print(color.violeta + "                  VARIABLES NO SIGNIFICATIVAS" + color.fin)
            tabla_1 = model_1.summary2().tables[1]
            filtrar_1 = tabla_1[tabla_1['P>|t|'] > p_value].sort_values('P>|t|',ascending =False)
            display(filtrar_1)
    
    
            lista_resultados.append([i, round(model_1.rsquared_adj,4), round(model_1.aic,2), round(model_1.bic,2)])

    
        df_mrl = pd.DataFrame(lista_resultados)
        print("\n\nSÍ CON LA ELIMINACIÓN DE CADA VARIABLE NO SIGNIFICATIVA")
        print(color.rojo + "DISMINUYE EL CRITERIO DE AIC-BIC Y SE INCREMENTA EL R2 ADJ " + color.fin)
        print("           ESTAMOS FRENTE A UN MEJOR MODELO !!!")
        df_mrl.columns = ['Vble Eliminada','R Cuadrado Ajustado', 'AIC', 'BIC']
        display(df_mrl)
        
        
    if plot == True:
        y_hat = model.predict(df.drop(X_drop, axis = 1))
        print(color.violeta + "\n\n                                           REAL VS PREDICCIÓN" + color.fin)
        plt.figure(figsize=(18,7))
        plt.plot(y, color='indigo', label = "Real" )
        plt.plot(y_hat, color='orange', label = "Predicción")
        plt.legend()
        plt.show()
        
        df_pred = df[[y_pred]]
        df_pred["Modelo"] = y_hat
        df_pred["Desvios"] = y - df_pred["Modelo"]
        df_pred["%Desvios"] = (y / df_pred["Modelo"]- 1)*100

        display(df_pred.head(5)) 
        print(f"\n\nEl modelo tiene un desvio promedio de: {round(np.abs(df_pred['%Desvios']).mean(),2)}")
        display(coeficientes)
    
    return model



def regresion_linealML(df,y_pred, test_size=.33,random_state=2054):
    
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir.

             
         Returs:
            
    """ 
    X, y = df.drop([y_pred], axis = 1), df[y_pred]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)

    model_1 = LinearRegression(fit_intercept = True, normalize = True).fit(X_train, y_train)  # Y = a + bX 
    model_2 = LinearRegression(fit_intercept = False, normalize = True).fit(X_train, y_train) # Y = bX
    model_3 = LinearRegression(fit_intercept = False, normalize = False).fit(X_train, y_train)

    modelos = ["REGRESION LINEAL 1(c/i y normalizado)","REGRESION LINEAL 2(s/i y normalizado)","REGRESION LINEAL 3(s/i y sin normalizar)"]


    for i, model in enumerate([model_1, model_2, model_3]):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        print(color.underline + "MODELO" + color.fin)
        print(color.azul + f"                              {modelos[i]}"+ color.fin)
        
    # Regression metrics
        r2 = metrics.r2_score(y_test, y_test_pred)
        mean_absolute_error = metrics.mean_absolute_error(y_test, y_test_pred) 
        mse = metrics.mean_squared_error(y_test, y_test_pred) 
        #mean_squared_log_error = metrics.mean_squared_log_error(y_test, y_test_pred)
        #median_absolute_error = metrics.median_absolute_error(y_test, y_test_pred)
        #explained_variance = metrics.explained_variance_score(y_test, y_test_pred)
        

        #print("-------------------------------")
        print('R2: ', round(r2,3))
        print('MAE: ', round(mean_absolute_error,3))
        print('MSE: ', round(mse,3))
        #print('RMSE: ', round(np.sqrt(mse),2))
        #print("-------------------------------")   
    
        plt.figure(figsize = (10,5))
        plt.subplot(1,2,2)
        sns.distplot(y_train - y_train_pred, bins = 20, label = 'train')
        sns.distplot(y_test - y_test_pred, bins = 20, label = 'test')
        plt.xlabel('Errores')
        plt.legend()
        
        ax = plt.subplot(1,2,1)
        ax.scatter(y_test,y_test_pred, s =40)
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes]
        ]  
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.xlabel('y (test)')
        plt.ylabel('y_pred (test)')   
        plt.tight_layout()
        plt.show()
    
    # COEFICIENTES
    print(color.azul +"\n\nEl valor que asumen los coeficientes en cada uno de los modelos es:"+ color.fin)
    coef = pd.DataFrame(df.columns).drop(0)
    coef = coef.rename(columns = {0:"Variables"})
    coef["Coeficientes_1"] = model_1.coef_
    coef["Coeficientes_2"] = model_2.coef_
    coef["Coeficientes_3"] = model_3.coef_
    display(coef)
    
    # INTERCEPTO
    print(color.azul +"\nEl valor que asumen el intercepto en cada uno de los modelos es:"+ color.fin)
    print("\nEl intercepto del modelo 1 es: ", round(model_1.intercept_ ,2))
    print("El intercepto del modelo 2 es: ", round(model_2.intercept_ ,2))
    print("El intercepto del modelo 3 es: ", round(model_3.intercept_ ,2))
    
    
    y_hat_ML1 = model_1.predict(X_train)
    y_hat_ML2 = model_2.predict(X_train)
    y_hat_ML3 = model_3.predict(X_train)
    
    modelo = pd.DataFrame(y_train)
    modelo["Modelo1_ML"] = y_hat_ML1
    modelo["Modelo2_ML"] = y_hat_ML2
    modelo["Modelo3_ML"] = y_hat_ML3
    
    modelo["Desvio1"] = modelo[y_pred] - modelo["Modelo1_ML"]
    modelo["Desvio2"] = modelo[y_pred] - modelo["Modelo2_ML"]
    modelo["Desvio3"] = modelo[y_pred] - modelo["Modelo3_ML"]
    
    modelo["%Desvios1"] = ( modelo[y_pred] / modelo["Modelo1_ML"]- 1 )*100
    modelo["%Desvios2"] = ( modelo[y_pred] / modelo["Modelo2_ML"]- 1 )*100
    modelo["%Desvios3"] = ( modelo[y_pred] / modelo["Modelo3_ML"]- 1 )*100
    
    #display(modelo.head(2))
    print(color.azul +"\n\n                                          DESVIO PROMEDIO\n"+ color.fin)
    print(f"El modelo 1 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios1']).mean(),3)}")
    print(f"El modelo 2 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios2']).mean(),3)}")
    print(f"El modelo 3 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios3']).mean(),3)}")





# A.3. AUTOREGRESSION INTEGRATED MOVING AVERAGE (ARIMA)
## Este modelo es un enfoque estadístico que combina la regresión lineal con los componentes de promedio móvil y diferenciación. ARIMA se utiliza para predecir valores futuros basándose en patrones identificados en los datos históricos.

def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window = 12).mean()
    MSTD = timeseries.rolling(window = 12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='teal',label='Original')
    mean = plt.plot(MA, color='navy', label='Rolling Mean')
    std = plt.plot(MSTD, color='cyan', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    
    
def dickey_fuller(timeseries):
    
    print('H0: No estacionaria. Tiene una raiz unitaria')
    print('H1: Estacionaria.\n')
    
    print('Results of Dickey-Fuller Test:')
    rtado = adfuller(timeseries)
    p_value = round(rtado[1],3)
    adf = round(rtado[0],2)
    alpha = 0.05
    
    print(f"P-value                 {p_value}")
    print(f"Test Statistic ADF      {adf}")
    for key, value in rtado[4].items():
        if adf > value:
            print("Para el valor crítico de (%s)  %.2f NO existe evidencia para rechazar HO" %(key, value))
        else:
            print("Para el valor crítico de (%s)  %.2f existe evidencia para rechazar HO" %(key, value))
                  
        
    if p_value > alpha:
        print(f"\nComo el p-value ({p_value}) > {alpha} alpha, y es posible que ADF {adf} > ( 1% y/o 5% y/o 10% ) \nNo existe evidencia suficiente para rechaza la Ho.")
        print(color.rojo + "\nLa series es NO estacionaria, tiene raiz unitaria y se debe transformar."+ color.fin)        
    else:
        print(f"\nComo el p-value {p_value} < {alpha} alpha, y es posible que ADF {adf} < ( 1% y/o 5% y/o 10% )")
        print(color.azul +"\nLa series es Estacionaria."+ color.fin)
    print('--------------------------------------------------')
    return 


def kpss_test(timeseries):
    print('H0: Es de tendencia estacionaria.')
    print('H1: No estacionaria.\n')
    
    print ('Results of KPSS Test:')
    rtado = kpss(timeseries, regression='c', nlags="auto")
    p_value = round(rtado[1],3)
    kpss_1 = round(rtado[0],2)
    alpha = 0.05
    
    print(f"P-value                  {p_value}")
    print(f"Test Statistic KPSS      {kpss_1}")
    for key,value in rtado[3].items():
        if kpss_1 < value:
            print("Para el valor crítico de (%s)  %.2f NO existe evidencia para rechazar HO" %(key, value))
        else:
            print("Para el valor crítico de (%s)  %.2f existe evidencia para rechazar HO" %(key, value))
        
    
    if p_value > alpha:
        print(f"\nComo el p-value ({p_value}) > {alpha} alpha, y es posible que {kpss_1} < ( 1% y/o 5% y/o 10% ) \nNo existe evidencia suficiente para rechaza la Ho.")
        print(color.azul +"\nLa series es al menos de tendencia estacionaria."+ color.fin)        
    else:
        print(f"\nComo el p-value {p_value} < {alpha} alpha, y es posible que  {kpss_1} > ( 1% y/o 5% y/o 10% )  .")
        print(color.rojo +"\nLa series es NO estacionaria, se debe transformar."+ color.fin) 
    print('-------------------------------------------------') 


def calculo_p_q(serie, d):
    data = []
    # Loop sobre el orden AR
    for p in range(13):
        #Loop sobre el orden MA
        for q in range(13):
            model = ARIMA(serie, order=(p,d,q))
            results = model.fit()
            data.append((p, q, round(results.aic,2), round(results.bic,2)))
    df_pq =pd.DataFrame(data,columns = ["p","q","AIC","BIC"])
    return df_pq 


def modelo_arima(serie, p, d, q):
    X = serie.values
    size = int(len(X) * 0.80)                  # El tamaño del set train es el 66% de la muestra
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    data = []
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q)) # Colocar los números que obtuvimos como p,d,q
        model_fit = model.fit()
        output = model_fit.forecast()
        y = output[0]
        predictions.append(y)
        real = test[t]
        history.append(real)
        residuos = real - y
    
        data.append((real, y, residuos))
    
    mse = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % mse)
    
    # Error Absoluto Medio : que tan lejos esta nuestra prediccion de los valores reales.
    mae = np.mean(np.abs(residuos))
    print(f'MAE: {round(mae,2)}')
    
    # R cuadrado
    r2 = r2_score(test, predictions)
    print(f"R2: {round(r2,2)}") 
    
    df_model = pd.DataFrame(data, columns = ["real", "pronóstico", "residuos"])
   
    
    # Grafico de valores esperados en comparación con las predicciones de pronostico continuo
    plt.figure(figsize=(18,7))
    plt.plot(test, color='indigo', label = "Test-Real" )
    plt.plot(predictions, color='orange', label = "Predicción")
    plt.legend()
    plt.show()
         
    return df_model



def residuos(vble, p, d, q, summary=False):
    model = ARIMA(vble, order = (p,d,q))
    results = model.fit()
    residuos = results.resid
    #modelo["residuals"] = residuos
    #print(round(residuos.describe(),1))
    #print("\nSí la media es distinta de cero, existe un sesgo en la prediccion.\n")
    if summary is False:
        return plt.show(results.plot_diagnostics(figsize=(15,12)))
    else:
        return results.summary()


    # 3.1 Autoregressive moving average models (ARMA)
     ## se utiliza cuando la serie de tiempo ya es estacionaria y no requiere de diferenciación.

    # 3.2 Seasonal Moving Average Integrated Autoregressive Models (SARIMA)
     ## Es útil cuando la serie de tiempo muestra patrones estacionales claros.

        
        
############ MULTIVARIATE PREDICTION

# A.4. VECTOR AUTOREGRESSION (VAR)
## Permite modelar la relación entre múltiples variables en un sistema simultáneamente. Cada variable se regresa a sus propios valores pasados y a los valores pasados de todas las demás variables.



# A.5. VECTOR ERROR CORRECTION (VEC)
## Para series de tiempo que exhiben relaciones de cointegración. Permite modelar tanto las relaciones a corto plazo como la restauración del equilibrio a largo plazo entre las variables.



       
############################### B. NEURAL NETWORKS
## Estos modelos tienen capacidad para capturar relaciones no lineales y patrones complejos en los datos

# B.0. RECURRENT NEURAL NETWORKS (RNNs)


# B.1. LONG-TERM MEMORY NEURAL NETWORKS (LSTMs)


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


