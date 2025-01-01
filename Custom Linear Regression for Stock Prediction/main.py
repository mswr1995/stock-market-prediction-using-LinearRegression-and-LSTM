import numpy as np
import pandas as pd
import os
import pandas_datareader as web
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import  train_test_split

import requests
from bs4 import BeautifulSoup

import datetime as dt

# Data Extraction from Wikipedia and Yahoo Fin
def savesp500tickers():
    result = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(result.text, "lxml")
    table = soup.find('table')
    tickers = []
    company = []

    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text
        stock = row.find_all('td')[1].text
        tickers.append(ticker)
        company.append(stock)

    with open("sp500ticker.pickle", "wb") as f:
        pickle.dump(tickers, f)

    with open("sp500company.pickle", "wb") as f:
        pickle.dump(company, f)

    for i in range(0, len(tickers)):
        print(tickers[i], "is the symbol for", company[i])
# savesp500tickers()


if not os.path.exists('stock_data'):
    os.makedirs('stock_data')
start= dt.datetime(2020,5,1)
end= dt.datetime(2020,7,2)

# Input for Stock name
x=input("Enter the Symbol of company: ")
df= web.DataReader(x,'yahoo',start,end)
df.to_csv('stock_data/stock.csv')

stock= pd.read_csv('stock_data/stock.csv')

stock.tail()

# # Single Variable Linear Regression
X = stock['Open'].values
y = stock['Adj Close'].values

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.21, random_state=0)

def Slope(a,b):
  n=len(a)
  slope=(n*np.sum(a*b)-np.sum(a)*np.sum(b))/(n*np.sum(a**2)-(np.sum(a))**2)
  return slope

# where x is equal to 0,
def Intercept(a,b):
  intercept = np.mean(b)-Slope(a,b)*np.mean(a)
  return intercept

# y = mx + b, m = slope b = intercept x = independent variable
def prediction1(a,b,x):
  pred1 = Slope(a,b)*x + Intercept(a,b)
  return pred1

def R_squared(pred,testing_set):
  a=pred
  b=testing_set
  ss_total=np.sum((b-np.mean(b))**2)
  ss_res=np.sum((b-a)**2)
  R_2=1-(ss_res/ss_total)
  return R_2

def Covariance(a,b):
  n=len(a)
  two_sum=np.mean(a*b)
  cov=two_sum/n-np.mean(a)*np.mean(b)
  return cov

def correlation_coeff(pred,testing_set):
  a=pred
  b=testing_set
  n=len(a)
  score=(n*np.sum(a*b)-np.sum(a)*np.sum(b))/np.sqrt((n*np.sum(a**2)-(np.sum(a))**2)*(n*np.sum(b**2)-(np.sum(b))**2))
  return score

covariance = Covariance(X,y)
print(covariance)

slope=Slope(Xtrain,ytrain)
intercept=Intercept(Xtrain,ytrain)
print(slope)
print(intercept)

prediction = prediction1(Xtrain,ytrain,Xtest)
print(Xtest)
print(prediction)

r2score = R_squared(prediction,ytest)
print(r2score)

corrcoef = correlation_coeff(prediction,ytest)
print(corrcoef)

linreg = slope * X + intercept

plt.scatter(X,y,marker='^',color='k',alpha=0.55)
plt.plot(X,linreg,color='R',linewidth=2)
plt.title('Single Variable Linear Regression')
plt.show()

# MAE
pred_list = [prediction]
test_list = [Xtest]
print('MAE: ' + str(mean_absolute_error(test_list, pred_list)))

# # Multi Variable Linear Regression
# X1 = stock['Open'].values
# train_size = int(0.78 * len(df))
# X1train = X1[:train_size]
# X1test = X1[train_size:]
#
# X2 = stock['High'].values
# train_size = int(0.79 * len(df))
# X2train = X2[:train_size]
# X2test = X2[train_size:]
# len(X2test)
#
# y = stock['Adj Close'].values
# train_size = int(0.79 * len(df))
# ytrain = y[:train_size]
# ytest = y[train_size:]
# len(ytest)
#
#
# def Slope1(a1, a2, b):
#     n = len(a1)
#     numerator1 = ((n * np.sum(a2 ** 2)) - (np.sum(a2) ** 2)) * ((n * np.sum(a1 * b)) - (np.sum(a1) * np.sum(b)))
#     numerator2 = ((n * np.sum(a1 * a2)) - (np.sum(a1) * np.sum(a2))) * ((n * np.sum(a2 * b)) - (np.sum(a2) * np.sum(b)))
#
#     denominator1 = ((n * np.sum(a1 ** 2)) - (np.sum(a1) ** 2)) * ((n * np.sum(a2 ** 2)) - (np.sum(a2) ** 2))
#     denominator2 = ((n * np.sum(a1 * a2)) - (np.sum(a1) * np.sum(a2))) ** 2
#
#     slope1 = (numerator1 - numerator2) / (denominator1 - denominator2)
#     return slope1
#
#
# def Slope2(a1, a2, b):
#     n = len(a1)
#     numerator1 = ((n * np.sum(a1 ** 2)) - (np.sum(a1) ** 2)) * ((n * np.sum(a2 * b)) - (np.sum(a2) * np.sum(b)))
#     numerator2 = ((n * np.sum(a1 * a2)) - (np.sum(a1) * np.sum(a2))) * ((n * np.sum(a1 * b)) - (np.sum(a1) * np.sum(b)))
#
#     denominator1 = ((n * np.sum(a1 ** 2)) - (np.sum(a1) ** 2)) * ((n * np.sum(a2 ** 2)) - (np.sum(a2) ** 2))
#     denominator2 = ((n * np.sum(a1 * a2)) - (np.sum(a1) * np.sum(a2))) ** 2
#
#     slope2 = (numerator1 - numerator2) / (denominator1 - denominator2)
#     return slope2
#
# def Intercept2(a1,a2,b):
#   intercept2 = np.mean(b)-Slope1(a1,a2,b)*np.mean(a1) - Slope2(a1,a2,b)*np.mean(a2)
#   return intercept2
#
#
# def prediction2(a1,a2,b,x1,x2):
#   pred2 = Slope1(a1,a2,b)*x1 + Slope2(a1,a2,b)*x2 + Intercept2(a1,a2,b)
#   return pred2
#
# def R_squared(pred,testing_set):
#   a=pred
#   b=testing_set
#   ss_total=np.sum((b-np.mean(b))**2)
#   ss_res=np.sum((b-a)**2)
#   R_2=1-(ss_res/ss_total)
#   return R_2
#
# def correlation_coeff(pred,testing_set):
#   a=pred
#   b=testing_set
#   n=len(a)
#   score=(n*np.sum(a*b)-np.sum(a)*np.sum(b))/np.sqrt((n*np.sum(a**2)-(np.sum(a))**2)*(n*np.sum(b**2)-(np.sum(b))**2))
#   return score
#
# slope1 = Slope1(X1train,X2train,ytrain)
# slope2 = Slope2(X1train,X2train,ytrain)
#
# intercept = Intercept2(X1train,X2train,ytrain)
# print(intercept)
#
# prediction= prediction2(X1train,X2train,ytrain,X1test,X2test)
# print(ytest)
# print(prediction)
#
# r2score2 = R_squared(prediction,ytest)
# print(r2score2)
#
# correcoef2 = correlation_coeff(prediction,ytest)
# print(correcoef2)
#
# linreg = slope1*X1  +slope2*X2+ intercept
#
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1, X2, y, c='r', marker='^')
# ax.scatter(X1,X2,linreg,c='blue',marker ='o')
# ax.set_xlabel('Open')
# ax.set_ylabel('High')
# ax.set_zlabel('Adj Close')
#
# plt.show()