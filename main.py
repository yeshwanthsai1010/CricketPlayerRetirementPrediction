import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import ANCHOR, PhotoImage, ttk,Canvas
import os
# Load the dataset
data = pd.read_csv('cricket_players_data.csv')
data.dropna()
del data['Player Name']
print(data)
# print(data.describe())
# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X['Injury History'])
# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)
# print(X['Total Test Runs'])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=data['Injury History'], random_state=0)
print(X_test['Injury History_Major Injuries'])
# Fit the linear regression model
model = LinearRegression().fit(X_train, y_train)
# Predict retirement age for the test set
X_train.to_csv("new.csv")
y_pred = model.predict(X_test)
# Create a scatterplot of the predicted vs actual values
sns.scatterplot(x=y_test, y=y_pred)
# Add a regression line to the plot
sns.regplot(x=y_test, y=y_pred)
# Add axis labels and a title to the plot
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
# Show the plot
# plt.show()
# Evaluate the model using mean squared error and Accuracy using Variance
mse = np.mean((y_pred - y_test) ** 2)
accuracy = 1 - mse/np.var(y_test)
print("Accuracy is: ",accuracy)
print("Mean squared error:", mse)
print("Model coefficients:", model.coef_)
def pred():
    name = txt.get()
    age = int(txt2.get())
    runs = int(txt3.get())
    wicks = int(txt4.get())
    Injury = txt5.get()
    fitness = txt6.get()
    bat = float(txt7.get())
    bowl = float(txt8.get())
    fiel = float(txt9.get())
    fl=False
    fh=False
    fm=False
    if(fitness=="Low"):
        fl=True
    elif(fitness=="Moderate"):
        fm=True
    else:
        fh=True
    In=False
    Imin=False
    Imaj=False
    if(Injury=="No"):
        In=True
    elif(Injury=="Minpr"):
        Imin=True
    else:
        Imaj=True
    hm = pd.DataFrame({'Age':[age],'Total Test Runs':[runs],'Total Test Wickets':[wicks],'Batting Average':[bat],'Bowling Average':[bowl],'Fielding Average':[fiel],'Fitness Level_Low':[fl],'Fitness Level_Moderate':[fm],'Injury History_Minor Injuries':[Imin],'Injury History_No Injuries':[In]})
    hm = pd.get_dummies(hm, drop_first=True)
    lm = model.predict(hm)
    txtres.config(text=lm[0])
# Interface
window = tk.Tk()
window.geometry("1700x720")
window.resizable(False,False)
window.title("Retirement Prediction")
message3 = tk.Label(window, text="Cricket Players Retirement Prediction from Test Career" ,fg="black",bg="#eddbda" ,width=60 ,height=1,font=('Times new roman', 29, ' bold '))
message3.place(x=2, y=10)

lbl = tk.Label(window, text="Enter Name:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl.place(x=50, y=80)
#
txt = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt.place(x=500, y=80)
#
lbl2 = tk.Label(window, text="Enter Age:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl2.place(x=50, y=120)
#
txt2 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt2.place(x=500, y=120)
#
lbl3 = tk.Label(window, text="Enter Total Test Runs:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl3.place(x=50, y=160)
#
txt3 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt3.place(x=500, y=160)
#
lbl4 = tk.Label(window, text="Enter Total Test Wickets:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl4.place(x=50, y=200)
#
txt4 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt4.place(x=500, y=200)
#
lbl5 = tk.Label(window, text="Enter Injury status({Major/Minor/No} Injury):",width=35 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 12, ' bold ') )
lbl5.place(x=50, y=240)
#
txt5 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt5.place(x=500, y=240)
#
lbl6 = tk.Label(window, text="Enter Fitness status({Low/Moderate/High}):",width=35 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 12, ' bold ') )
lbl6.place(x=50, y=280)
#
txt6 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt6.place(x=500, y=280)
#
lbl7 = tk.Label(window, text="Enter Batting Average:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl7.place(x=50, y=320)
#
txt7 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt7.place(x=500, y=320)
#
lbl8 = tk.Label(window, text="Enter Bowling Average:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl8.place(x=50, y=360)
#
txt8 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt8.place(x=500, y=360)
#
lbl9 = tk.Label(window, text="Enter Fielding Average:",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbl9.place(x=50, y=400)
#
txt9 = tk.Entry(window,width=40,fg="black" ,font=('Times new roman', 15, ' bold '))
txt9.place(x=500, y=400)
#
predi = tk.Button(window, text="Push to Predict!", command=pred  ,fg="black"  ,bg="#f7f1a1"  ,width=18 ,activebackground = "white" ,font=('Times new roman', 14, ' bold '))
predi.place(x=300, y=460)
#
frameres=tk.Label(window,width=65,height=26,background="grey")
frameres.place(x=1100,y=120)
#
s = '''***Disclaimer***
The Output of this Application are purely \n dependent on requested Factors/attributes \n as the Retirement of a player will depend \n on Various Factors. Please do not get \n Offended if the Predicted Age is \n Wrong, \n Thank You!!. '''
lblres = tk.Label(frameres,text="Predicted Retirement Age",width=30 ,height=1  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lblres.place(relx=0.1,rely=0.1)
txtres = tk.Label(frameres,width=30,fg="black" ,font=('Times new roman', 14, ' bold '))
txtres.place(relx=0.1, rely=0.2)
lbldis = tk.Label(frameres,text=s,width=30 ,height=8  ,fg="black"  ,bg="#f7f1a1" ,font=('Times new roman', 14, ' bold ') )
lbldis.place(relx=0.1,rely=0.3)
# message = tk.Label(window, text="Enter Test Carrer Total wickets" ,bg="#f7f1a1" ,fg="black"  ,width=50,height=1, activebackground = "#3ffc00" ,font=('Times new roman', 20, ' bold '))
# message.place(x=7, y=450)

# lbl3 = tk.Label(frame1, text="Attendance",width=37 ,fg="black"  ,bg="#f7f1a1"  ,height=1 ,font=('Times new roman', 20, ' bold '))
# lbl3.place(x=100, y=115)
window.mainloop()