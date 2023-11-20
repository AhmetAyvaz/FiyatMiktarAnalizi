import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
from scipy import stats

rawdata = pd.read_csv("C:/Users/USER/Desktop/Project/csv/online_retail_II.csv")
data = rawdata.copy()

data.drop(data.columns[[0]],axis=1,inplace=True)

#? Veri Temizliği:
# InvoiceNo: Invoice number Nominal.A 6-digit integral number uniquely assigned to each transaction 
# If this code starts with the letter 'c', it indicates a cancellation.
Invoice_isnull = data["Invoice"].isnull().sum()
error1_index = data[~data["Invoice"].str.isdigit()].index 
data.drop(data.index[error1_index],inplace=True)
error2_index = data[(data["Invoice"].str.len() != 6)].index 
data.drop(data.index[error2_index],inplace=True)
error3_index = data[data["Invoice"].str.startswith("c")].index 
data.drop(data.index[error3_index],inplace=True)

# StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
StockCode = data["StockCode"].isnull().sum() 
error4_index_raw = data[(data["StockCode"].str.len()!=5) | (~data["StockCode"].str.isdigit())].index.tolist()
error4_index = []
for i in error4_index_raw:
    if i >= 532619:
        continue
    else:
        error4_index.append(i)
data.drop(data.index[error4_index],inplace=True)

# Quantity: The quantities of each product (item) per transaction. Numeric.	
Quantity_isnull = data["Quantity"].isnull().sum()
index = np.where([data["Quantity"] <= 0])[1].tolist()
data.drop(data.index[index],axis=0,inplace=True)


# InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
data["InvoiceDate"] = data["InvoiceDate"].dt.date

# Country: Country name. Nominal. The name of the country where a customer resides.
Country_isnull = data["Country"].isnull().sum()

Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1

highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR

highestLimitOutlier = data[data["Price"] >= highestLimit].index.tolist()
lowestLimitOutlier = data[data["Price"] <= lowestLimit].index.tolist()

outlier = highestLimitOutlier + lowestLimitOutlier
data.loc[outlier,"Price"] = data["Price"].mean()

Q1 = data["Quantity"].quantile(0.25)
Q3 = data["Quantity"].quantile(0.75)
IQR = Q3 - Q1

highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR

highestLimitOutlier = data[data["Quantity"] > highestLimit].index.tolist()
lowestLimitOutlier = data[data["Quantity"] < lowestLimit].index.tolist()
outlier = highestLimitOutlier + lowestLimitOutlier

data.loc[outlier,"Quantity"] = data["Quantity"].mean()

data["Total Price"] = (data["Quantity"] * data["Price"])
index = np.where([data["Total Price"] <= 0])[1].tolist()
data.drop(data.index[index],axis=0,inplace=True)

priceOfDate = data.groupby("InvoiceDate").agg({"Total Price":"sum"})["Total Price"]
quantityOfDate = data.groupby("InvoiceDate").agg({"Quantity":"sum"})["Quantity"]
liste1 = []
liste2 = []
for i in priceOfDate:
     liste1.append(i) 
for i in quantityOfDate:
    liste2.append(i)
df = pd.DataFrame({"priceOfDate":pd.Series(liste1),"quantityOfDate":pd.Series(liste2)})

kurtosis = df["priceOfDate"].kurtosis() # -1.5 < 0.75 < 1.5
skew = df["priceOfDate"].skew() # -1.5 < 0.87 < 1.5
shapiro = stats.shapiro(df["priceOfDate"]) # 0.0000000087
ks = stats.kstest(df["priceOfDate"],cdf="norm",args=(df["priceOfDate"].mean(),df["priceOfDate"].std())) # 0.000690

kurtosis = df["quantityOfDate"].kurtosis()
skew = df["quantityOfDate"].skew()
shapiro = stats.shapiro(df["quantityOfDate"])
ks = stats.kstest(df["quantityOfDate"],cdf="norm",args=(df["quantityOfDate"].mean(),df["quantityOfDate"].std()))


corr = df.corr(method="spearman")

productTop = data.groupby("Description").agg({"Total Price":"sum","Quantity":"sum","Price":"sum"}).sort_values(ascending=False,by="Total Price")

liste3 = []
liste4 = []
for i in productTop["Total Price"]:
    liste3.append(i)
for i in productTop["Quantity"]:
    liste4.append(i)

df["Total Price"] = pd.Series(liste3)
df["Quantity"] = pd.Series(liste4)
transfer = data[["Price","Description","Total Price","Quantity"]].sort_values(ascending=False,by="Price")

liste1 = []
liste2 = []
liste3 = []
liste4 = []
for i in transfer["Description"]:
    liste1.append(i)
for i in transfer["Price"]:
    liste2.append(i)
for i in transfer["Quantity"]:
    liste3.append(i)
for i in transfer["Total Price"]:
    liste4.append(i)

dataset = pd.DataFrame({"Description":pd.Series(liste1),"Price":pd.Series(liste2),"Quantity":pd.Series(liste3),"Total Price":pd.Series(liste4)})

tempData = dataset.groupby("Price").agg({"Quantity":"sum","Price":"sum",})
liste5 = []
liste6 = []
for i in tempData["Quantity"]:
    liste5.append(i)
for i in tempData["Price"]:
    liste6.append(i)
actualData = pd.DataFrame({"Quantity":pd.Series(liste5),"Temp Price":pd.Series(liste6)})
actualData["Unit Price"] = actualData["Temp Price"] / actualData["Quantity"]
actualData.drop("Temp Price",axis=1,inplace=True)

skew = np.sqrt(actualData["Unit Price"]).skew()

Q1 = actualData["Unit Price"].quantile(0.25)
Q3 = actualData["Unit Price"].quantile(0.75)
IQR = Q3 - Q1

highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR

highestOutlier = actualData[actualData["Unit Price"] > highestLimit]["Unit Price"]
lowestOutlier = actualData[actualData["Unit Price"] < lowestLimit]

shapiro = stats.shapiro(np.sqrt(actualData["Unit Price"]))

skew = actualData["Quantity"].skew()

Q1 = actualData["Quantity"].quantile(0.25)
Q3 = actualData["Quantity"].quantile(0.75)
IQR = Q3 - Q1

highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR

highestOutlier = actualData[actualData["Quantity"] > highestLimit]["Quantity"]
lowestOutlier = actualData[actualData["Quantity"] < lowestLimit]

shapiro = stats.shapiro(np.sqrt(actualData["Quantity"]))

data = np.sqrt(actualData)

corr = pg.corr(data["Unit Price"],data["Quantity"],method="spearman")

sns.heatmap(data.corr(method="spearman"),annot=True,cmap="YlGnBu")
plt.show()