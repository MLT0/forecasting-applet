# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:04:46 2018

@author: MTanny
"""

import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import datetime
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import inspect

data      = pd.read_csv("C:\\Users\\MTANNY\\Documents\\Python Scripts\\testdata.csv",dtype={'Date':str})
features  = pd.read_csv("C:\\Users\\MTANNY\\Documents\\Python Scripts\\features.csv",dtype=str)
modelspec = pd.read_csv("C:\\Users\\MTANNY\\Documents\\Python Scripts\\modelspec.csv",dtype=str)
periods   = pd.DataFrame({'Text':['Year','Month','Week','Day'],'Value':['A','M','W','D']})


time_var  = "Date"        
groupvar = ["Category"]        
measure_var  = ["Value"]
response_var = "Value"
period   = "M"

# get the INITIAL data
startdata = data[:]
startdata = startdata.groupby(pd.DatetimeIndex(startdata[time_var]).to_period(period))[measure_var].sum()
startdata = startdata.reindex(pd.DatetimeIndex(start=min(startdata.index).to_timestamp(),
                                               end=(max(startdata.index)+1).to_timestamp(),
                                               freq=period).to_period(period)).fillna(0)

sourcelines = pd.DataFrame(inspect.getsourcelines(GradientBoostingRegressor)[0])

def generateFeatureFunction(name,prefix,function):
    return('''    
def %s(self,data):
    a = data[:]
    newcol = "%s" + response_var
    a[newcol] = a[response_var]%s
    return a[[newcol]], newcol
    '''%(name,prefix,function))

class GetFeatures():
    for i in features.index:
        exec(generateFeatureFunction(features.iat[i,0],features.iat[i,1],features.iat[i,2]))
        
        
def addFeatures(df,varcheckbox,fun):
    if varcheckbox.get()==1:
        df = pd.concat([df,fun(df)[0]],axis=1)[:]
    return(df)

#df = startdata[:]

def simplePredict(data,model):
    for i in data.index:
        data.ix[i,[response_var]] = model.predict(data.drop([response_var],axis=1).ix[i].reshape(1,-1))
    return(data)

def processData(df,*args):
    #get Test Period
    NtestPeriod = df.shape[0]-(varSliderTestTrain.get())
    #Save actual value
    actual = df[[response_var]]
    #Set space for Predictions      
    df.ix[(varSliderTestTrain.get()):,[response_var]] = 0
    
    #Get features as chosen
    for i in features.index:
        df = addFeatures(df,eval("varCheckBox%s"%(features.iat[i,1])),eval("GetFeatures().%s"%(features.iat[i,0])))       

       
    dfTrain = df[:-NtestPeriod] if NtestPeriod!=0 else df[:]
    dfdnaTrain = dfTrain.dropna()[:]
    
    #Select Models and Parameters
    model_parameters = {i.split('_', 1)[1]:eval(i+".get()") for i in modParam[modParam.str.contains(varSelectModel.get())]}
    
    '''    
    paramTable = pd.DataFrame({'model_name':modParam.apply(lambda x:x.split('_', 1)[0]).str[4:],
                               'parameter':modParam.apply(lambda x:x.split('_', 1)[1]),
                               'value':modParam.apply(lambda x:eval(x+".get()"))})    
    paramTable[paramTable['model_name']==varSelectModel.get()].set_index(['parameter']).value.to_dict()

    type(PVARGBM_n_estimators.get())
    '''
    #model_parameters = {'n_estimators': 20, 'max_depth': 5}    
    
    model = eval("%s(**model_parameters)"%(modelspec[modelspec['model_name']==varSelectModel.get()]['model_function'].unique()[0]))
    model.fit(dfdnaTrain.drop([response_var], axis=1),dfdnaTrain[response_var])
    
    if varCheckBoxSimplePredict.get()!=0:
        dfdnaTrain = df.dropna()[:]
    
    if NtestPeriod!=0 and varCheckBoxSimplePredict.get()==0:         
        for i in range(-NtestPeriod,0):
            df.ix[i,[response_var]] = model.predict(df.drop([response_var],axis=1).ix[i].reshape(1,-1))
            df = df[[response_var]]
            for j in features.index:
                df = addFeatures(df,eval("varCheckBox%s"%(features.iat[j,1])),eval("GetFeatures().%s"%(features.iat[j,0])))
             
        dfdnaTrain = simplePredict(dfdnaTrain,model)
        dfdnaTrain = dfdnaTrain.append(df[-NtestPeriod:])

    else:  
        #Get Simple-Prediction (Direct, fast, but inaccurate)
        dfdnaTrain = simplePredict(dfdnaTrain,model)
        
    importances = pd.DataFrame(model.feature_importances_).set_index(dfdnaTrain.drop([response_var], axis=1).columns)
    
    
    
    res = pd.concat([actual,dfdnaTrain[response_var]],axis=1)
    res.columns = ['Actual','Predicted']    
       
    
    #MAD, MSE, MAPE, etc
    error = pd.DataFrame({'MSE':np.mean((res['Actual'] - res['Predicted'])**2),
                          'MAD':np.mean(np.abs(res['Actual'] - res['Predicted'])),
                          'MAPE':np.mean(np.abs((res['Actual'] - res['Predicted'])/res['Actual']))}, index=[0]).transpose()

    
    return(res,importances,error)

def changeModelParameterSet(*args):
    if varSelectModel.get()=="GBM":
        RFparamFrame.forget()
        GBMparamFrame.pack(side="top")       
    elif varSelectModel.get()=="RF":
        GBMparamFrame.forget()
        RFparamFrame.pack(side="top")

    
def rebootData(data):
    startdata = data[:]
    startdata = startdata.groupby(pd.DatetimeIndex(startdata[time_var]).to_period(varSelectPeriod.get()))[measure_var].sum()
    startdata = startdata.reindex(pd.DatetimeIndex(start=min(startdata.index).to_timestamp(),
                                                   end=(max(startdata.index)+1).to_timestamp(),
                                                   freq=varSelectPeriod.get()).to_period(varSelectPeriod.get())).fillna(0)    
    return(startdata)

def updatePlot(*args):
    
    startdata = rebootData(data)  
    df = processData(startdata) 
       
    ax.clear()
    ax.plot(df[0].set_index(df[0].index.to_timestamp().to_pydatetime()))
    ax.axvline(x=(min(startdata.index)-1+IOBJsliderTT.get()).to_timestamp())   
    
    ay.clear()
    ay.bar(range(1,(df[1].shape[0])+1),df[1][0])    
    
    
    az.clear()
    az.bar(range(1,(df[2].shape[0])+1),df[2][0]) 
   
    canvas.draw()

def TimePeriodSliderResetValue(*args):
    startdata = rebootData(data)
    IOBJsliderTT.set(startdata.shape[0])


def updateTimePeriodSlider(*args):
    if varCheckBoxPause.get()==0:
        startdata = rebootData(data)
        IOBJsliderTT.configure(from_=1, to=(startdata.shape[0]))    
        

def createFeatureCheckBox(master,name,prefix,command):
    return('''
varCheckBox%s = tk.IntVar(value=1)
IOBJcheckbox%s = tk.Checkbutton(%s, text="%s", variable=varCheckBox%s,command=%s)
IOBJcheckbox%s.pack(side="top") 
    '''%(prefix,prefix,master,name,prefix,command,prefix))

def createRadioButton(master,name,value,variable,command,prefix,position):
    return(
prefix+"rb%s=tk.Radiobutton(%s,text='%s', variable=%s, value='%s',command=%s)"%(name,master,name,variable,value,command)+"\n"+
prefix+"rb%s.pack(side='%s',anchor='n')"%(name,position) 
    )

'''
def createSliderwName(name,value,variable,command,prefix,position):
    return(
prefix+"rb%s=tk.Scale(master,text='%s', variable=%s, value='%s',command=%s)"%(name,name,variable,value,command)+"\n"+
prefix+"rb%s.pack(side='%s')"%(name,position) 
    )
'''

 



def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

def pauseUpdate(*args):
    if varCheckBoxPause.get()==1:            
        for i in varCom.index:
            exec("%s.configure(command=False)"%(varCom[i]))
        
    if varCheckBoxPause.get()==0:            
        for i in varCom.index:
            if 'IOBJ' in varCom[i]:
                a = Icommand
            elif 'DOBJ' in varCom[i]:
                a = Dcommand
            elif 'MODL' in varCom[i]:
                a = Mcommand
            elif 'PARM' in varCom[i]:
                a = Icommand
            exec("%s.configure(command=%s)"%(varCom[i],a))    
        

def test(*args):
    0



master = tk.Tk()


varSliderTestTrain       = tk.IntVar(value=startdata.shape[0])
varSelectPeriod          = tk.StringVar()
varSelectModel           = tk.StringVar()
varCheckBoxPause         = tk.IntVar(value=0)
varCheckBoxSimplePredict = tk.IntVar(value=0)

Icommand = "updatePlot"
Dcommand = "combine_funcs(updatePlot, updateTimePeriodSlider,TimePeriodSliderResetValue)"
Mcommand = "combine_funcs(updatePlot, changeModelParameterSet)"

col1Frame = tk.Frame(master)
col2Frame = tk.Frame(master)

# COLUMN 1 ########################
modelFrame  = tk.Frame(col1Frame)


for i in modelspec['model_name'].unique():
    exec(createRadioButton("modelFrame",i,i,"varSelectModel",Mcommand,"MODL","left"))
MODLrbGBM.select()
modelFrame.pack(side="top")


GBMparamFrame  = tk.Frame(col1Frame)

learning_rateFrame = tk.Frame(GBMparamFrame)
tk.Label(learning_rateFrame, text="learning_rate").pack(side="left")
PVARGBM_learning_rate = tk.DoubleVar(value=0.1)
PARMGBMsliderlearning_rate = tk.Scale(learning_rateFrame, from_=0.01, to=0.5, resolution=0.01, command=updatePlot, variable=PVARGBM_learning_rate, orient = 'horizontal')
PARMGBMsliderlearning_rate.pack(side="left") 
learning_rateFrame.pack(side="top")

n_estimatorsFrame = tk.Frame(GBMparamFrame)
tk.Label(n_estimatorsFrame, text="n_estimators").pack(side="left")
PVARGBM_n_estimators = tk.IntVar(value=100)
PARMGBMslidern_estimators= tk.Scale(n_estimatorsFrame, from_=10, to=500, resolution=1, command=updatePlot, variable=PVARGBM_n_estimators, orient = 'horizontal')
PARMGBMslidern_estimators.pack(side="left") 
n_estimatorsFrame.pack(side="top")

max_depthFrame = tk.Frame(GBMparamFrame)
tk.Label(max_depthFrame, text="max_depth").pack(side="left")
PVARGBM_max_depth = tk.IntVar(value=3)
PARMGBMslidermax_depth= tk.Scale(max_depthFrame, from_=1, to=20, resolution=1, command=updatePlot, variable=PVARGBM_max_depth, orient = 'horizontal')
PARMGBMslidermax_depth.pack(side="left") 
max_depthFrame.pack(side="top")




RFparamFrame  = tk.Frame(col1Frame)

n_estimatorsFrame = tk.Frame(RFparamFrame)
tk.Label(n_estimatorsFrame, text="n_estimators").pack(side="left")
PVARRF_n_estimators = tk.IntVar(value=100)
PARMRFslidern_estimators= tk.Scale(n_estimatorsFrame, from_=10, to=500, resolution=1, command=updatePlot, variable=PVARRF_n_estimators, orient = 'horizontal')
PARMRFslidern_estimators.pack(side="left") 
n_estimatorsFrame.pack(side="top")

min_samples_leafFrame = tk.Frame(RFparamFrame)
tk.Label(min_samples_leafFrame, text="min_samples_leaf").pack(side="left")
PVARRF_min_samples_leaf = tk.DoubleVar(value=0.1)
PARMRFslidermin_samples_leaf= tk.Scale(min_samples_leafFrame, from_=0.01, to=0.5, resolution=0.011, command=updatePlot, variable=PVARRF_min_samples_leaf, orient = 'horizontal')
PARMRFslidermin_samples_leaf.pack(side="left") 
min_samples_leafFrame.pack(side="top")


modParam = pd.Series(dir())[pd.Series(dir()).str.contains('PVAR')]



GBMparamFrame.pack(side="top")


# COLUMN 2 ########################
utilsFrame   = tk.Frame(col2Frame)
periodFrame  = tk.Frame(col2Frame)
sliderFrame  = tk.Frame(col2Frame)
featureFrame = tk.Frame(col2Frame)

checkboxSimplePredict = tk.Checkbutton(utilsFrame, text="SimplePredict", variable=varCheckBoxSimplePredict,command=processData)
checkboxSimplePredict.pack(side="left") 

checkboxPause = tk.Checkbutton(utilsFrame, text="Pause", variable=varCheckBoxPause,command=combine_funcs(pauseUpdate,updatePlot,updateTimePeriodSlider))
checkboxPause.pack(side="left") 
varCom = pd.Series(dir())[pd.Series(dir()).str.contains('OBJ')]

for i in periods.index:
    exec(createRadioButton("periodFrame",periods.iat[i,0],periods.iat[i,1],"varSelectPeriod",Dcommand,"DOBJ","left"))
DOBJrbMonth.select()

IOBJsliderTT = tk.Scale(sliderFrame, from_=1, to=(startdata.shape[0]),orient = 'horizontal', resolution=1,command=updatePlot,variable=varSliderTestTrain)
IOBJsliderTT.pack(side="top")

for i in features.index:
    exec(createFeatureCheckBox("featureFrame",features.iat[i,0],features.iat[i,1],Icommand))

utilsFrame.pack(side="top")
periodFrame.pack(side="top")
sliderFrame.pack(side="top")
featureFrame.pack(side="top") 
###################################

# CANVAS ##########################

df = processData(startdata)[0]
fig = Figure()
ax = fig.add_subplot(1,2,1)
ay = fig.add_subplot(2,2,2)
az = fig.add_subplot(2,2,4)
ax.plot(df.set_index(df.index.to_timestamp().to_pydatetime()))
ax.axvline(x=(min(startdata.index)-1+startdata.shape[0]).to_timestamp())

canvas = FigureCanvasTkAgg(fig,master=master)

### Packing all the frames ######

col2Frame.pack(side='left')
col1Frame.pack(side='left')
canvas.get_tk_widget().pack(side='right',fill='both', expand=1)
canvas.draw()

tk.mainloop()
