import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('/content/cars 2.csv')

df.drop(['Car','Model'],axis=1,inplace=True)
df.head()
x=df[['Volume','Weight']]
y=df['CO2']

from sklearn.preprocessing import StandardScaler # type: ignore
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split # type: ignore
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
r2=r2_score(y_test,y_predict)
print('mae:',mse)
print('mae:',mae)
print('r2:',r2)

plt.scatter(y_test,y_predict)
plt.show()

newdata=[[2300,170]]
new_data=sc.transform(newdata)
pred=model.predict(new_data)
print(pred)