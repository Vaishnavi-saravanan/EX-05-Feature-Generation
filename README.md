# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE:
# DATA.CSV
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# ENCODING.CSV
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
# TITANIC.CSV
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT:
# DATA CSV

# Initial Dataset:
![out1](https://user-images.githubusercontent.com/118541897/232683506-e9827280-5e6c-4abd-8be4-f117d09cdebb.png)

# Binary Encoding:
![out2](https://user-images.githubusercontent.com/118541897/232683517-c4757b16-f5dc-44da-8081-40a54006b7c2.png)

# Encoded Dataset:
![out3](https://user-images.githubusercontent.com/118541897/232683531-0168a847-9c4f-4d35-abc7-eca7a5e01c02.png)

# Data Scaling using MinMaxScaler:
![out04](https://user-images.githubusercontent.com/118541897/232683562-b62ba568-bba6-4393-921c-d69bc928ed99.png)

# Data Scaling using StandardScaler:
![out4](https://user-images.githubusercontent.com/118541897/232683579-d7574f37-a258-4708-87b0-c35a3b0bcf82.png)

# Data Scaling using MaxAbsScaler:
![out5](https://user-images.githubusercontent.com/118541897/232683590-e8c31afa-f5ab-4283-b5a2-e1dc35fac0bb.png)

# Encoding.csv :
# Initial Dataset:
![out6](https://user-images.githubusercontent.com/118541897/232684567-a0702e12-fb05-40f9-86f6-60f8d6237f57.png)

# Binary Encoding:
![out7](https://user-images.githubusercontent.com/118541897/232684747-f81da8d7-aa54-4798-98c8-41ef06650900.png)

# Encoded Dataset:
![out8](https://user-images.githubusercontent.com/118541897/232685082-5a2a4147-495d-4a41-9da0-0561c9380c2e.png)

# Data Scaling using MinMaxScaler:
![out9](https://user-images.githubusercontent.com/118541897/232685152-853436df-1930-4b1b-ab30-a152c7968d55.png)

# Data Scaling using MaxAbsScaler:
![out10](https://user-images.githubusercontent.com/118541897/232685184-052d8e11-bf3e-4b74-8eb0-48c98fc87ccd.png)

Data Scaling using RobustScaler:
![out11](https://user-images.githubusercontent.com/118541897/232685223-1be86b90-e7b7-4374-a694-0bd7b2e40d8c.png)

# Titanic.csv :
# Initial Dataset:
![out12](https://user-images.githubusercontent.com/118541897/232685333-cbea7b66-6967-4f94-b26d-c01066a2395e.png)

# Data cleaning before encoding:
![out13](https://user-images.githubusercontent.com/118541897/232685408-e94f21d1-ac52-431c-ab10-74c659e731e5.png)

# Cleaned Dataset:
![out14](https://user-images.githubusercontent.com/118541897/232685642-405351b0-c86b-4a84-9d7e-f1f078cabd98.png)

# Binary Encoding:
![out15](https://user-images.githubusercontent.com/118541897/232685666-74c4145c-1aa9-4d79-8bf7-1bcf9db7446d.png)

# Encoded Dataset:
![out16](https://user-images.githubusercontent.com/118541897/232685689-039e27ed-fc25-4897-ba4b-39586d4a7374.png)

# Data Scaling using MinMaxScaler:
![out17](https://user-images.githubusercontent.com/118541897/232685716-1ce27ead-1ca5-4789-8745-7dd551976641.png)

# Data Scaling using StandardScaler:
![out18](https://user-images.githubusercontent.com/118541897/232685742-2068b554-0970-4891-ae99-05b9d2525038.png)

# Data Scaling using MaxAbsScaler:
![out19](https://user-images.githubusercontent.com/118541897/232685759-3342da86-41f9-4e63-ad99-bb6777c79b24.png)

# Data Scaling using RobustScaler:
![out20](https://user-images.githubusercontent.com/118541897/232685816-4932edbc-4da4-4119-a7dd-bcbce258da0f.png)

# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
