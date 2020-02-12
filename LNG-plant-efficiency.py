# -----------------------------------------------------------
# Calculation of overall efficiency in LNG plant 
#
# (C) 2020 Jerson Rodas Alarcon, Lima, Peru
# Released under GNU Public License (GPL)
# email jerson017@hotmail.com / jrodas19@gmail.com 
# -----------------------------------------------------------

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
sns.set()

# Load data
#Feed gas molar composition in % (C1, C2, C3, nC4, iC4, nC5, iC5, C6, CO2, N2)
#LNG molar composition in % (C1, C2, C3, nC4, iC4, nC5, iC5, C6, CO2, N2)
#Feed gas flow (m3/h)
#Feed gas density at line conditions (kg/m3)
#LNG rundown temperature (°C)
#LNG mass flow (kg/h)
df=pd.read_csv('datasets/hypothetical-plant-data.csv')
df['Date'] = pd.to_datetime(df['Date'], format = '%d-%m-%y %H:%M')
df=df.set_index('Date')
# Compositions must be normalizaded (sum must be 100%)
feed_comp = df[['feed-C1', 'feed-C2', 'feed-C3', 'feed-iC4', 'feed-nC4','feed-iC5', 'feed-nC5', 'feed-C6', 'feed-N2', 'feed-CO2']]/100
lng_comp = df[['LNG-C1','LNG-C2','LNG-C3','LNG-iC4', 'LNG-nC4','LNG-iC5','LNG-nC5','LNG-C6','LNG-N2',]]/100

# Molecular weights and heating from GPA-2145-03 (Gas Processors Association)
molec_weights = {'C1':16.0420, 'C2':30.0690, 'C3':44.0960,'nC4':58.1220, 'iC4':58.1220, 'nC5':72.1490,'iC5':72.1490,  'C6':86.1750, 'N2':28.0135, 'CO2':44.0100}
heating_values = {'C1':55.5760, 'C2':51.9520, 'C3':50.3700,'nC4':49.5470,'iC4':49.3890, 'nC5':49.0460,'iC5':48.9500,  'C6':48.7170, 'N2':0.0000, 'CO2':0.0000}

# Heating value calculation for feed (NG) and product (LNG)       
NGHHV = []      # Inicialization of feed heating value
LNGHHV = []     # nicialization of product heating value
XM = [] 

for i in range (len(df)):

    total1=sum(feed_comp.iloc[i]*list(molec_weights.values())*list(heating_values.values()))
    sum1=sum(feed_comp.iloc[i]*list(molec_weights.values()))
    NGHHV.append(total1/sum1)

    total2=sum(lng_comp.iloc[i]*(list(molec_weights.values())[:9])*(list(heating_values.values())[:9]))
    sum2=sum(lng_comp.iloc[i]*(list(molec_weights.values())[:9]))
    LNGHHV.append(total2/sum2)       

    mol_weight=sum(feed_comp.iloc[i]*list(molec_weights.values()))
    XM.append(mol_weight)  

df['NG-HHV MJ/kg']=NGHHV
df['LNG-HHV MJ/kg']=LNGHHV
df['XM'] = XM

# LNG Density Calculation
# The revised Klosek-McKinley method was used to calculate LNG density, for more information go to ASTM-D4784

## Molar Volume calculation for each component at LNG temperature
mol_vol=pd.read_csv('datasets/molar_volum.csv', sep=';')   #Load  volume molar for pure components dataset
X=mol_vol['T (C)'].values.reshape(-1, 1)         
regressor1_7=[]        # Molar volumes are calculated using interpolation (Regression)          
for i in range(7):
    Y=mol_vol.iloc[:,i+1].values.reshape(-1, 1)  
    r = LinearRegression( )     
    r.fit(X,Y)
    regressor1_7.append(r)

b=mol_vol.iloc[:,[0,8]].dropna()
X8=b.iloc[:,0].values.reshape(-1, 1)
Y8=b.iloc[:,1].values.reshape(-1, 1)
regressor8 = LinearRegression()
regressor8.fit(X8,Y8)

X9=mol_vol['T (C)'].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
X9_ = poly.fit_transform(X9)
Y9=mol_vol['N2'].values.reshape(-1, 1)
regressor9 = LinearRegression()
regressor9.fit(X9_,Y9)

molarvol = []
for i in range (len(df)):
    T=df['LNG-temperature'][i] 
    a=[]
    for i in range(7):
        a.append(regressor1_7[i].predict(np.array([T]).reshape(-1, 1)))
    a.append(regressor8.predict(np.array([T]).reshape(-1, 1)))
    a.append(regressor9.predict(poly.fit_transform(np.array([T]).reshape(-1, 1))))
    molarvol.append(a)  
df['Molar Volume']=molarvol

XVM=[]
for j in range (len(df)):
    sum = 0
    for i in range(9):
        sum = sum + df['Molar Volume'][j][i]*lng_comp.iloc[j][i]
    XVM.append(sum[0][0])
df['XVM']=XVM


## K1 factor calculation at LNG temperature
K1=pd.read_csv('datasets/k1.csv', sep=';')      #Load  K1 data set
K1_unpivoted = K1.melt(id_vars=['Temp'], var_name='MW', value_name='k1')
XK1=K1_unpivoted[['Temp', 'MW']]
poly = PolynomialFeatures(degree=2)
XK1_ = poly.fit_transform(XK1)
YK1=K1_unpivoted['k1']
regressor = LinearRegression()      # K1 is calculated using interpolation (Regression)
regressor.fit(XK1_,YK1)
k1 = []
for i in range (len(df)):
    T=df['LNG-temperature'][i] 
    M=df['XM'][i]
    k1.append(0.001*regressor.predict(poly.fit_transform([[T, M]]).tolist())[0])
df['k1']=k1

## K2 factor calculation at LNG temperature
K2=pd.read_csv('datasets/k2.csv', sep=';')      #Load  K2 data set
K2_unpivoted = K2.melt(id_vars=['Temp'], var_name='MW', value_name='k2')
XK2=K2_unpivoted[['Temp', 'MW']]
poly = PolynomialFeatures(degree=2)
XK2_ = poly.fit_transform(XK2)
YK2=K2_unpivoted['k2']
regressor = LinearRegression()
regressor.fit(XK2_,YK2)
k2 = []
for i in range (len(df)):
    T=df['LNG-temperature'][i] 
    M=df['XM'][i]
    k2.append(0.001*regressor.predict(poly.fit_transform([[T, M]]).tolist())[0])
df['k2']=k2


## Revised Klosek-McKinley equation 
df['Klosek-McKinley-Density'] = df['XM'] / ( df['XVM'] - (df['k1']+ (df['k2']-df['k1'])*(lng_comp['LNG-N2']/0.0425))*lng_comp['LNG-C1'] )
 
# Energy In (feed) calculation
energy_in = (df['feed-flow']*df['feed-density'])*df['NG-HHV MJ/kg']

# Energy Out (product) calculation
f=((df['Klosek-McKinley-Density']/460)**0.5)
energy_out = df['LNG mass flow']*f*df['LNG-HHV MJ/kg']
#Efficiency
df['Overall Efficiency'] = 100*energy_out.values/energy_in.values

#Plot
df['Overall Efficiency'].plot()
plt.title('Overall Efficiencny LNG Plant')
plt.ylabel('%')
plt.legend()
plt.show()


