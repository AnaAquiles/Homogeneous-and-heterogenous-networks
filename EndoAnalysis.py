# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:12:43 2019

@author: kirex
"""



# CÓDIGO DE ANALISIS DE SERIES DE TIEMPO DE CÉLULAS ENDÓCRINAS

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import signal
from scipy import stats 
from matplotlib import cm
import pandas as pd
import seaborn as sns 


filename="YourFile"                                                            
data=np.loadtxt(filename + ".csv",delimiter=',')

# Cuando se tengan tratamientos del mismo tamaño de imagen,descomentar con
#   (CTRL y 1) y correr:
      
# Cuando NO se tenga el mismo número de imágenes por estímulo, correr: 
datos = np.array([data[280:420]]) #en data:, colocar el número correspondiente al límite de la primera adquisición, seguido de :
datos = np.swapaxes(datos,0,1)
    
i = 140  # límite donde la actividad basal se termina
#Normalization respect the baseline: 

def NormF(datos):
    baseline=np.amin(datos[:,:,:i],-1)[:,:,None] # este valor tiene que ser igual al númerototal de imágenes a analizar por tratamiento 
    return datos/baseline
    
# Debleach of activity and fluorescence normalization using linear regression of the basal activity for each condition

def detrend(datos,window=i):#mismo valor que en baseline
    x=np.arange(0,window)
    x = x[None,:]*np.ones((datos.shape[-2],1))
    x=np.ravel(x)
    slopes=[]
    intercepts=[]
    for dat in datos:
        y = np.ravel(dat[:,:window])
        slope,inter,_,_,_=stats.linregress(x,y)
        slopes.append(slope)
        intercepts.append(inter)
    slopes=np.array(slopes)
    intercepts=np.array(intercepts)
    t=np.arange(0,datos.shape[-1])
    trends=np.array((intercepts)[:,None] + np.array(slopes)[:,None] * t[None,:])
    return datos - trends[:,None,:]
     
b,a = signal.bessel(3,0.1,btype='lowpass') #grado del filtrado 0.1
datosfilt=signal.filtfilt(b,a,datos,axis=-1)
datosNorm=detrend(NormF(datos))
datosNormFilt=(NormF(datosfilt)) 
dt=0.2  # tiempo de adquisición de cada frame
time=np.arange(0,dt*datosNorm.shape[-1],dt) 

#%%

#  Raster plot de la actividad 

from matplotlib import pyplot 
 
series = [] 
 
for ini in range(0,datos.shape[1]): 
    for j in range(len(datos)): 
        series.append(datosNormFilt[j,ini,:]) 
 
 
series = np.array(series)
series = pd.DataFrame(series)
 
pyplot.matshow(series, interpolation=None, aspect = 'auto', cmap='bone') 
pyplot.colorbar() 
 
 
pyplot.xlabel('Time acquisition') 
pyplot.ylabel('Cells') 
pyplot.yticks(np.arange(0,894,894)) #colocar el número total de células del raster 
pyplot.xticks(np.arange(300,0,))  # colocar en el primer valor, el # de imágenes del tratamiento
pyplot.show()

#%%%

#           Grafica con señal global del o los tratamientos

#       SELECCIONA EL NÚMERO i= x DE TRATAMIENTO QUE QUIERES GRAFICAR 
#          si sólo tienes 1 tratamiento, dejar i=0


i= 0
 #Número de tratamiento
plt.figure(3)
plt.clf()
plt.subplot(321)
plt.plot(datos[i,:,:].T)

plt.subplot(323)
plt.plot(NormF(datos)[i,:,:].T)

plt.subplot(325)
plt.plot(detrend(NormF(datos))[i,:,:].T)
    

plt.subplot(322)
plt.plot(datosfilt[i,:,:].T)

plt.subplot(324)
plt.plot(NormF(datosfilt)[i,:,:].T)

plt.subplot(326)
plt.plot(detrend(NormF(datosfilt))[i,:,:].T)

#%%

#     Calculo de los valores de ABC, Max/Min y Tasa de decaimiento

signals = (datos[0,:,:])
cells = np.arange(0,894,1)
#                                       si lo quieres calcular de todas tus imágenes, dejar :#total
#                                       si lo quieres hacer en un rango referente, colocar #inicio:#final


# Valores de abc POR simpson para cada célula 

Abc = simps(signals,axis=-1) 

# valores de la tasa de decaimiento para cada célula 

for i in range(len(signals)):
    tau = np.polyfit(time,np.log(signals[i]),1)   

Max = np.max(signals,axis=-1) 
Min = np.min(signals, axis=-1) 

#%%

# Valores de Frecuencia y Amplitud 

i = 0

fftabsdata = (datosNormFilt[i,:,:]) 

for n in range(0,len(fftabsdata)):
    f,pspec =  signal.welch(fftabsdata, fs = 3.3, window ='hanning',
                        nperseg= 300, noverlap = 3.3//2, 
                        nfft= None, detrend = 'linear',return_onesided= True,
                        scaling='spectrum')
# Valores de amplitud 

Amplitude = np.array(pspec)

meanFreq_part = np.mean(Amplitude, axis=0)
sdFreq_part = np.std(Amplitude, axis=0) 
varFreq_part = np.var(Amplitude,axis = 0)


#Amplitude = np.array(pspec)
#meanAmp = np.mean(Amplitude,axis = 0)
#sdAmp = np.std(Amplitude,axis = 0) 
#varAmp = np.var(Amplitude,axis = 0)
#    
##    Amplitude[Amplitude<(meanAmp + 0.5*sdAmp)]=0
#Amp = np.mean(Amplitude, axis=1)

plt.figure(3)
plt.clf()

plt.subplot(121)
plt.plot(f,Amplitude.T, c='b')
plt.plot(f,meanFreq_part, c='r')
plt.plot(f,varFreq_part
         , c='k')
#plt.xlim(0,0.05)
#plt.legend('Global PSD','mean Global PSD', 'SD Global PSD')
plt.ylabel('Power (w/Smooth)')
plt.xlabel('Frequency (Hz)')
plt.title('Global PowerSpectra')


Abc = simps(pspec)
meanAbc = np.mean(Abc)
sdAbc = np.std(Abc)

plt.subplot(122)
plt.hist(meanFreq_part, bins=10, )
plt.ylabel('Frequency')
plt.xlabel('Power values')
plt.title('Global Power values')


#%%

#  Correlación de actividad 
datosNorm=detrend(NormF(datos))


def SurrogateCorrData(datos,N=1000): #Número de veces en las que se generará las matrices aleatorizadas #NO MOVER
    fftdatos=np.fft.fft(datos,axis=-1)
    ang=np.angle(fftdatos)
    amp=np.abs(fftdatos)
    #Cálculo de la matriz de correlación de los datos aleatorizados
    CorrMat=[]
    for i in range(N):
        angSurr=np.random.uniform(-np.pi,np.pi,size=ang.shape)
        angSurr[:,70:]= - angSurr[:,70:0:-1] 
        angSurr[:,70]=0       #tenemos que colocar únicamente el valor correspondiente a la mitad de las imágenes de nuestro estudio
        
        fftdatosSurr=np.cos(angSurr)*amp + 1j*np.sin(angSurr)*amp
    
        datosSurr=np.real(np.fft.ifft(fftdatosSurr,axis=-1)) #arroja la valores reales de los datos aleatorizados
        spcorr2,pval2=stats.spearmanr(datosSurr,axis=1)
        CorrMat.append(spcorr2)
        
    CorrMat=np.array(CorrMat)
    return CorrMat
  

SCM=SurrogateCorrData(datosNorm[i])     

#Calculate the standart desviation and mean of SCM=SurrogateCorrData
meanSCM=np.mean(SCM,0)
sdSCM=np.std(SCM,0)



# GRÁFICOS DE LAS MATRICES DE CORRELACIÓN     v  



#   Ploteo de las matrices de correlación considerando la desviación estándar (2) de la distribución de la matriz aleatorizada

spcorr,pval=stats.spearmanr(datosNorm[i],axis=1) 
#spcorr[pval>=0.0001]=0


#Filtro de la matriz original, que tomará como 0 a los valores abs de la correlación que sean menores a 2SD del promedio de SCM, 
 #          Cambiamos a tres derviaciones estándar
spcorr[np.abs(spcorr)<(meanSCM + 2*sdSCM)]=0

#np.savetxt(filename +"01spcorr.csv", spcorr, delimiter=',')

#      Gráficas de correlación 


plt.figure(4)
plt.clf()

 
plt.subplot(231)
plt.plot(datosNorm[i].T)

plt.subplot(232)
plt.imshow(spcorr,interpolation='none',cmap='inferno',vmin=-1,vmax=1)
plt.colorbar()
plt.grid(False)    

plt.subplot(233)
plt.hist(spcorr.ravel(),bins=50)

plt.subplot(234)
plt.plot(SCM[i])

plt.subplot(235)
plt.imshow(np.std(SCM,0),interpolation='none',cmap='viridis')
#plt.imshow(spcorr2,interpolation='none',cmap='jet')

plt.grid(False)    

plt.subplot(236)
plt.hist(SCM[:,5,8],bins=50)



#%%

#     Correlación estable

# Seleccionamos valores significativos de correlación 

spcorr2=np.tril(spcorr)
z = np.where((spcorr2>0.35) & (spcorr2<0.9)) 
zz=[]

#usando zip para iterar las parejas 
for i1,i2 in zip(z[0],z[1]):
    zz.append((i1,i2, spcorr2[i1,i2]))
#def remove_duplicates(i):
#    return list (set(i))
zz=np.array(zz)  

# Cargar los valores de las coordenadas

#%%
coors= np.loadtxt('Coord.csv', delimiter= ',') # colocar el nombre del archivo con la extensión.csv

#     SACAMOS LA CANTIDAD DE CONEXIONES QUE TIENE CADA CÉLULA 


binsp = 1*(np.abs(spcorr2>0.45) & (spcorr2<0.9)).astype(float)

conex = (sum(binsp).astype(float)) 

Nodes = (np.where(conex!=0)) 

Conex = (conex[conex!=0])

 #CONEXIONES DE TODAS LAS CÉLULAS RESPONSIVAS
 
Nodes = np.array(Nodes).T.astype(float) 
Nodos= np.vstack((Nodes.T[0],Conex)).T  ####Número de nodo más el número de conexiones que tiene
Nod = Nodos[:,0] 

Total=[]      
     
for i in range(0,len(Nod)):
    Total.append(coors[int(Nod[i]),:])

Total = np.array(Total) #Coordenadas del número total de células


np.savetxt(filename +"zz.csv", zz, delimiter=',')
np.savetxt(filename +"Total.csv", Total, delimiter=',')

#%%

# Scatter map de los nodos con p < 0.45 # de conexiones por nodo

plt.style.use('seaborn-whitegrid')


x = Total[:,0]
y = Total[:,1]

plt.figure(6)
plt.scatter(x,y, s = Conex, c= Conex, cmap = "seismic",alpha =0.5) 
plt.show()


#%%

#   Gráfica de la Red

plt.style.use('seaborn-whitegrid')


x = Total[:,0]
y = Total[:,1]

plt.plot([x],[y],'k.',ms=8)
for link in zz:
    plt.plot((x[link[0]],x[link[1]]),(y[link[0]],y[link[1]]),'-',linewidth=0.8,
             c=cm.seismic(link[2]/2+0.3),lw=np.abs(link[2])*1)

##plt.colorbar()
plt.grid(False)
#plt.colorbar()

#%%

#  Histogramas de probabilidad de conexión

plt.style.use('seaborn-whitegrid')

fig, axes = plt.subplots(
                         )

plt.hist(Nod,bins=50, facecolor='k', normed= True) 
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(Conex))


u = np.mean(lnspc) 
v = np.var(lnspc)
K = stats.kurtosis(lnspc)

m, s = stats.norm.fit(Conex) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Norm") # plot it

ag,bg,cg = stats.gamma.fit(Conex)  
pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
plt.plot(lnspc, pdf_gamma, label="Gamma")

ab,bb,cb,db = stats.beta.fit(Conex)  
pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
plt.plot(lnspc, pdf_beta, label="Beta")

#axes.set_ylabels('Probability')
#axes.set_xlabels('Degree')


axes.axis([0,150,0,0.01])


plt.show() 
#%%


### MÉTRICAS LOCALES

import networkx as nx

G= nx.Graph()

zz_=np.delete(zz,[2],axis=1)

zzTotal_list = zz_.tolist()
       
E = G.add_edges_from(zzTotal_list)


Nnodos = nx.number_of_nodes(G)
Density = nx.density(G)
Cluster = nx.average_clustering(G)
Assortativity = nx.degree_assortativity_coefficient(G)
ShortPath = nx.average_shortest_path_length(G)   
Connected = np.array(nx.connected_component_subgraphs(G))

plt.style.use('seaborn-whitegrid')
nx.draw(G,node_size=100,node_color='tomato',edge_color='gray')
