

# AB Testing Project

***İş Problemi***

Facebook kısa süre önce mevcut maximum bidding adı verilen teklif
verme türüne alternatif olarak yeni bir teklif türü olan average bidding’i
tanıttı.
Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test
etmeye karar verdi ve averagebidding’in, maximumbidding’den daha
fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak
istiyor.

* Maximum Bidding: Maksimum teklif verme
* Average Bidding: Average teklif verme

***Veri Seti Hikayesi***

bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların
gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen
kazanç bilgileri yer almaktadır.
Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri
ab_testing.xlsx excelinin ayrı sayfalarında yer almaktadır.

***Değişkenler***

* Impression – Reklam görüntüleme sayısı
* Click – Tıklama : Görüntülenen reklama tıklanma sayısını belirtir.
* Purchase – Satın alım : Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.
* Earning – Kazanç : Satın alınan ürünler sonrası elde edilen kazanç

***GÖREV - 1***

A/B testinin hipotezini tanımlayınız.
"""

#Uyarı mesajlarını kapatmak için:
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore",category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.7f' % x)

# Veri Setinin Hazırlanması:

def load_dataset(xls_file, sheetname=False):
  if sheetname:
    return pd.read_excel(xls_file+".xlsx", sheet_name = sheetname)
  else:
    pd.read_excel(xls_file +".xlsx")

df_control = load_dataset("ab_testing" , sheetname="Control Group")
df_test = load_dataset("ab_testing", sheetname="Test Group")

df_control.head()

df_test.head()

#####################################
df_control = pd.read_excel(r"C:\Users\ASUS\Desktop\DSMLBC-8\WEEK_4\ab_testing_veri\ab_testing.xlsx", sheet_name="Control Group")
df_test= pd.read_excel(r"C:\Users\ASUS\Desktop\DSMLBC-8\WEEK_4\ab_testing_veri\ab_testing.xlsx", sheet_name="Test Group")

##############################################################
df_test.shape

df_control.shape

#  Describing The Data (Verileri tanımlamak)

def check_df(dataframe, head=5):
    """
    This Function returns:
        - shape : The dimension of dataframe.
        - size : Number of elements in the dataframe.
        - type : The data type of each variable.
        - Column Names : The column labels of the DataFrame.
        - Head : The first "n" rows of the DataFrame.
        - Tail : The last "n" rows of the DataFrame.
        - Null Values : Checking if any "NA" Value is into DataFrame
        - quantile : The Basics Statistics

    Parameters
    ----------
    dataframe : dataframe
        Dataframe where the dataset is kept.
    head : int, optional
        The function which is used to get the first "n" rows.

    Returns
    -------

    Examples
    ------
        import pandas as pd
        df = pd.read_csv("titanic.csv")
        print(check_df(df,10))
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Size #####################")
    print(dataframe.size)
    print("##################### Type #####################")
    print(dataframe.dtypes)
    print("############### Column Names ####################")
    print(dataframe.columns)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################## Null Values ##################")
    print(dataframe.isnull().values.any())
    print("################## Quantiles ####################")
    print(dataframe.quantile(q=[0, 0.25, 0.50, 0.75,1]))

check_df(df_control)

# Grafikle de inceleyelim: Sayısal değişkenler için boxplot(kutu grafiği)

sns.boxplot(df_control["Impression"]);

sns.boxplot(df_control["Click"], color="pink");

sns.boxplot(df_control["Purchase"], color="green");

sns.boxplot(df_control["Earning"], color="orange");
plt.show()

check_df(df_test)

# Uç değer görünüyor, baskılama yöntemi ile çözebiliriz ancak şimdilik göz ardı edebiliriz
# (Amaç reklam çıktılarını ölçmek olduğu için göz ardı edelim)
sns.boxplot(df_test["Impression"], color="blue");

sns.boxplot(df_test["Click"], color="pink");

sns.boxplot(df_test["Purchase"], color="green");

sns.boxplot(df_test["Earning"], color="orange");
plt.show()
# Test grubunda Impression değişkenine outlier değerler mevcut, baskılama yöntemi ile çözümleyelim:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Baskılama işlemini uygulayalım:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_thresholds(df_test, "Impression")
replace_with_thresholds(df_test, "Impression")

# Outlier değerleri baskıladık:

sns.boxplot(df_test["Impression"], color="blue");

# İki veri setini birleştirelim: öncelikle group diye bir kolon ekleyerek control/test ayrımını belirtelim:
 
df_control["Group"] = "Control_Group"
df_control.head()

df_test["Group"] = "Test_Group"
df_test.head()

df = pd.concat([df_control, df_test],axis=0)
df.head()

df["Group"].value_counts()

# Veri setini günlük değerlendirerek gün değişkeni ekleyelim:
# indexi değişken gibi kullanabiliriz:
# Bunu yapmaktaki sebebimiz 40 günü incelediğimiz için gün bazında ilerlemiş oluyoruz.
df.reset_index(inplace=True)
df.head()

df["index"] = df["index"]+1  #İndexlerin 1'den başlamasını istediğimiz için +1 ekledik.
df.head()

df.rename(columns={"index":"Day"}, inplace=True)
df.head()

df["Day"].max()

df["Group"].unique()

df.columns

# Yeni Değişkenlerin Hesaplanması: 

# Görüntüleyenlerin ne kadarı tıkladı?

df["Conversion_Rate"] = df["Click"] * 100 / df["Impression"] #reklamı görüntülediler ve bu görüntülemelerin % kaçı reklama girdi tıkladı.
df.head()

# Tıklayanların ne kadarı ürün satın aldı?

df["Success_Rate"] = df["Purchase"] * 100 / df["Click"]

#df.drop("index",axis=1,inplace=True)

df.head()

#  Betimsel istatistikleri genel olarak kontrol edelim:

df.groupby("Group").agg({"Impression":"mean", \
                 "Click":"mean", \
                 "Purchase":"mean", \
                 "Earning":"mean", \
                 "Conversion_Rate":"mean", \
                 "Success_Rate":"mean"})

# Günlük Reklam Görüntüleme sayısı Karşılaştırması

sns.lineplot(x="Day", y="Impression",hue="Group",data=df)
plt.title("Günlük Reklam Görüntüleme Sayısı Karşılaştırması")
plt.legend(loc="upper right", bbox_to_anchor=(0.6, -0.2))
plt.figure(figsize=(15,15));
plt.show()
# Reklam görüntülenme sayılarında bazı günler hariç, ağırlıklı olarak test grubunda daha yüksek görünüyor

# Günlük Tıklama Sayılarının Karşılaştırması

sns.lineplot(x="Day", y="Click",hue="Group",data=df)
plt.title("Günlük Tıklama Sayılarının Karşılaştırması")
plt.legend(loc="upper right", bbox_to_anchor=(0.6, -0.2))
plt.figure(figsize=(20,20));


# Tıklama sayıları ise kontrol grubunda çok daha yüksek görünmektedir

# Tıklama oranlarına bakalım:

sns.lineplot(x="Day", y="Conversion_Rate",hue="Group",data=df)
plt.title("Günlük Tıklama Oranlarının (Click/Impression) Karşılaştırması")
plt.legend(loc="upper right", bbox_to_anchor=(0.6, -0.2))
plt.figure(figsize=(20,20));


# Oransal olarak kontrol ettiğimizde de, control grubunda görüntüleme yapanların daha büyük bir kısmında aynı zamanda tıklama da gerçekleşmektedir

# Günlük Satın Alma Sayılarının Karşılaştırması

sns.lineplot(x="Day", y="Purchase",hue="Group",data=df)
plt.title("Günlük Satın Alma Sayısı Karşılaştırması")
plt.legend(loc="upper right", bbox_to_anchor=(0.6, -0.2))
plt.figure(figsize=(20,20));

# Satın alma sayılarında her iki grup içinde dalgalanma görülmektedir

# Günlük Satın Alma Oranlarının Karşılaştırması

sns.lineplot(x="Day", 
             y="Purchase",
             hue="Group",
             data=df,  
             markers=True,
             dashes=False)

plt.title("Günlük Satın Alma Oranlarının (Purchase/Click) Karşılaştırması")
plt.legend(loc="upper right", bbox_to_anchor=(0.6, -0.2))
plt.figure(figsize=(20,20));
plt.show()
# Satın alma sayılarında her iki grup içinde dalgalanma görülmektedir

"""***Adımlar***


1. Hipotezlerin Kurulması
2. Varsayım Kontrolü
  - 1. Normallik Varsayımı
  - 2. Varyans Homojenliği
3. Hipotezin Uygulanması
  - p-value < 0.05 ise Ho red.
  - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
  - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
* Not:
 - Normallik sağlanmıyorsa mannwhitneyu testi,  Varyans homojenliği sağlanmıyorsa parametrik test arguman girilir (equal_var=False)

***GÖREV - 2***

Hipotez testini gerçekleştiriniz. Çıkan
sonuçların istatistiksel olarak anlamlı olup
olmadığını yorumlayınız.

**GÖREV - 3**

Hangi testi kullandınız, sebeplerini belirtiniz.

**Performans açısından aşağıdaki metrikler ele alınmıştır:**

- Tıklama Sayısı 
- Satın Alma Sayısı
- Tıklama Oranı (Click/Impression)
- Başarı Oranı (Purchase/Click)

***Tıklama Sayısı***
"""

###### Click #############

#1. Hipotezin Kurulması

# H0: Maximum bidding ile average bidding için, tıklamalar arasında anlamlı bir farklılık yoktur.
# H1: Maximum bidding ile average bidding için, tıklamalar arasında anlamlı bir farklılık vardır.


# H0: M1 = M2
# H1: M1!= M2 

print(df.loc[df["Group"] == "Control_Group", ["Click"]].mean())
print(df.loc[df["Group"] == "Test_Group", ["Click"]].mean())

?shapiro

###### Click #############

# 2. Normallik Varsayımı

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control_Group", "Click"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro( df.loc[df["Group"] == "Test_Group", "Click"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# p_value >0.05 olduğunda Ho reddedilmez, yani normallik varsayımı sağlanmaktadır

?levene

###### Click #############

# 3. Varyans Homojenligi Varsayımı

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir


test_stat, pvalue = levene(df.loc[df["Group"] == "Control_Group", "Click"],
                           df.loc[df["Group"] == "Test_Group", "Click"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p_value < 0.05 olduğundan Ho reddedilir, yani varyanslar homojen değildir.
# Ancak normallik sağlandığı için argüman eklenerek (equal_var=False) ttest uygulanabilir

###### Click #############
print(df.loc[df["Group"] == "Control_Group", "Click"].mean())
print(df.loc[df["Group"] == "Test_Group", "Click"].mean())


test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "Control_Group", "Click"],
                              df.loc[df["Group"] == "Test_Group", "Click"],
                              equal_var=False)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# < 0.05 olduğundan H0 RED. click sayılarında anlamlı bir farklılık VARDIR!!!

# Kontrol grubu için tıklama sayıları çok daha yüksektir.
# Tekniğe karar vermek için diğer değişkenleri değerlendireceğiz.

"""***Satın Alma Sayısı***"""

###### Satın Alma Sayısı #############

# 1. Hipotez Testi: 

# H0: Maximum bidding ile average bidding için, satın alma sayıları arasında anlamlı bir farklılık yoktur.
# H1: Maximum bidding ile average bidding için, satın alma sayıları arasında anlamlı bir farklılık vardır.

# H0: M1 = M2
# H1: M1!= M2
 
print(df.loc[df["Group"] == "Control_Group", ["Purchase"]].mean())
print(df.loc[df["Group"] == "Test_Group", ["Purchase"]].mean())

###### Satın Alma Sayısı #############

# 2. Normallik Varsayımı

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control_Group", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro( df.loc[df["Group"] == "Test_Group", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# >0.05 olduğundan reddedilemez! Normallik varsayımı sağlanmaktadır.

######  Satın Alma Sayısı #############
 
# 3. Varyans Homojenliği Varsayımı

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["Group"] == "Control_Group", "Purchase"],
                           df.loc[df["Group"] == "Test_Group", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


 # Ho > 0.05 olduğundan Ho hipotezi reddedilmez, yani varyansalar homojendir.

?ttest_ind

print(df.loc[df["Group"] == "Control_Group", "Purchase"].mean())
print(df.loc[df["Group"] == "Test_Group", "Purchase"].mean())



test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "Control_Group", "Purchase"],
                              df.loc[df["Group"] == "Test_Group", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# > 0.05 olduğundan Ho reddedilmez, yani satın alma sayılarında anlamlı bir farklılık yoktur

"""***Tıklama Oranı (Click/Impression)***"""

nobs = df.loc[df["Group"] == "Control_Group",:].shape[0]
nobs

#type(nobs)

df.head()

# 1. Hipotez Testi:  (İlk olarak satın alma için bakalım)

# H0: Maximum bidding ile average bidding için tıklama oranları arasında anlamlı bir farklılık yoktur.
# H1: Maximum bidding ile average bidding için tıklama oranları arasında anlamlı bir farklılık vardır.

# H0: M1 = M2
# H1: M1!= M2 
print(df.loc[df["Group"] == "Control_Group", ["Conversion_Rate"]].mean())
print(df.loc[df["Group"] == "Test_Group", ["Conversion_Rate"]].mean())

"""*Proportions Z-test ile Günlük tıklama oranları arasında anlamlı bir farklılık var mı gözlemleyelim:*"""

# Hipotezin Kurulması:


# H0: Maximum bidding ile average bidding için tıklama oranları arasında anlamlı bir farklılık yoktur.
# H1: Maximum bidding ile average bidding için tıklama oranları arasında anlamlı bir farklılık vardır.

# Click Rate proportional z-test
df_control.rename(columns={"Impression": "Impression_C","Click": "Click_C", "Purchase": "Purchase_C"}, inplace=True)
df_test.rename(columns={"Impression": "Impression_T","Click": "Click_T", "Purchase": "Purchase_T"}, inplace=True)



df_ = pd.concat([df_control ,df_test], axis = 1 )[["Impression_C","Impression_T","Click_C","Click_T", "Purchase_C", "Purchase_T"]]
df_.head()

# Kontrol grubunda tıklanma oranı daha yüksek:

print(np.mean(df_["Click_C"] /df_["Impression_C"]))
print(np.mean(df_["Click_T"] /df_["Impression_T"]))

# Test grubunda tıklandıktan sonra satın alma sayısı daha yüksek:

print(np.mean(df_["Purchase_C"] /df_["Click_C"]))
print(np.mean(df_["Purchase_T"] /df_["Click_T"]))

# basari_sayisi = np.array([300, 250])
# gozlem_sayilari = np.array([1000, 1100])
# proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

# Satırlarda iterativ bir işlem uygularak her bir gün özelinde kontrol ve test grup arasında farklılık var mı bakalım:
df_.apply(lambda x: proportions_ztest(count = [x["Click_C"], x["Click_T"]],  \
                                      nobs =  [x["Impression_C"], x["Impression_T"]]), axis = 1)

# Satırlarda iterativ bir işlem uygularak her bir gün özelinde kontrol ve test grup arasında farklılık var mı bakalım:

df_click_proptest = pd.DataFrame.from_records(df_.apply(lambda x: proportions_ztest(count = [x["Click_C"], x["Click_T"]], 
                                      nobs =  [x["Impression_C"], x["Impression_T"]]), axis = 1), 
                                      columns=['test_stat', 'p_value'])

df_click_proptest

df_click_proptest[df_click_proptest["p_value"]<0.05]

df_click_proptest[df_click_proptest["p_value"]<0.05].shape

df_click_proptest[df_click_proptest["p_value"]<0.05].shape[0]

df_click_proptest.sort_values("p_value",ascending=False).head(5)

# Satın alma oranlarını inceleyelim:

df_purch_proptest = pd.DataFrame.from_records(df_.apply(lambda x: proportions_ztest(count = [x["Purchase_C"], x["Purchase_T"]], 
                                      nobs =  [x["Click_C"], x["Click_T"]]), axis = 1), 
                                      columns=['test_stat', 'p_value'])

df_purch_proptest[df_purch_proptest["p_value"]<0.05].shape[0]

df_purch_proptest[df_purch_proptest["p_value"]>=0.05].sort_values("p_value",ascending=False)

"""**Fonksiyonlaştıralım:**"""

# Veri seti hazırlık :

df_control.rename(columns={"Impression": "Impression_C","Click": "Click_C", "Purchase":"Purchase_C"}, inplace=True)
df_test.rename(columns={"Impression": "Impression_T","Click": "Click_T", "Purchase":"Purchase_T"}, inplace=True)

 # Testte kullanılacak alanları tek bir dataframe üzerinde birleştirelim:
df_ = pd.concat([df_control ,df_test], axis = 1 )[["Impression_C","Impression_T",
                                                   "Click_C","Click_T",
                                                   "Purchase_C","Purchase_T"]]

# Proportion_z_test fonksiyonu:

def prop_ztest(dataframe, Numerator1, Numerator2, Demominator1, Demominator2, ratio=0.5):
  df_proptest = pd.DataFrame.from_records(dataframe.apply(lambda x: proportions_ztest(count = [x[Numerator1], x[Numerator2]], 
                                      nobs =  [x[Demominator1], x[Demominator2] ]), axis = 1), 
                                      columns=['test_stat', 'p_value'])
  if df_proptest[df_proptest["p_value"]<0.05].shape[0] >=  dataframe.shape[0] * ratio:
    print("Kontrol ve Test Grubu arasında arasında anlamlı bir farklılık vardır")
  else:
    print("Kontrol ve Test Grubu arasında arasında anlamlı bir farklılık yoktur")

prop_ztest(df_ ,"Click_C","Click_T", "Impression_C","Impression_T", 0.1)

# 30 /40

prop_ztest(df_, "Purchase_C", "Purchase_T",  "Click_C","Click_T", 0.5 )

prop_ztest(df_, "Purchase_C", "Purchase_T",  "Click_C","Click_T" ,0.8)

"""**Sonuc:**
* Tıklama sayılarında (Click) istatistiksel olarak anlamlı farklılık görülmüş olup,  kontrol grubunda tıklanmalar daha yüksektir. Bu durumda maximum bidding yöntemi ile devam edilmelidir.

* Satın alma sayılarında her iki grup için anlamlı bir farklılık vardır  Bu durum da maximum bidding yöntemini desteklemektedir.

* Tıklama ve satın alma oranları için de anlamlı bir farklılık görülmüştür (günlük oranlar üzerinden değerlendirildiğinde gözlem sayısının %50 ve üzerinde bu farka rastlanmıştır), amaç tıklama oranını arttırmak ise Max bidding ile devam edilebilir. Ancak ana odağımız satışa dönme oranı ise Avg Bidding seçilebilir

"""
