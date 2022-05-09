---
title: Aykırı Değerlerle Başa Çıkmak
date: 2022-05-09
categories: [ML]
tags: []     # TAG names should always be lowercase
---

# Aykırı Değerlerle Başa Çıkmak

![image.png](https://miro.medium.com/max/1400/1*xzpXzg2TB9uoXRUdIXBe6Q.jpeg)

Bu yazımızda veri ön işlemenin önemli adımlarından biri olan aykırı değer sorununu python uygulaması ile nasıl çözebileceğimizi ele alacağız.

###### "Verileriniz kötüyse, makine öğrenimi araçlarınız işe yaramaz." (harvard business review)

Genel olarak, makine öğrenmesi öğrenenler için 🙂, veri ön işlemenin sonunda verilerinizi modele fit etmek en heyecan verici kısımdır. Ne yazık ki makine öğrenmesi çalışmalarının büyük bir kısmı verilerin model için hazırlanmasından oluşur. Aykırı değerlerle uğraşmak, özellik mühendisliğinin ve veri ön işlemenin önemli bir parçasıdır. Bunun için aykırı değerlerle nasıl başa çıkabileceğimize bir göz atalım.

#### Aykırı Değerler ve Yaklaşım Türleri

Aykırı değerler, veri kümesinin genel yapısını bozan ve doğrusal modellerde sorunlara neden olan verilerdir. Aykırı değerlerle başa çıkmak için farklı yöntemler denenebilir.

##### Sektör Bilgisi

Bunlardan biri, sektör bilginize dayanarak, uğraştığınız sektöre ait bir veri setinde sahip olduğunuz verilerin eşik değerlerini belirlemektir. (Kiralık evler verisi ile uğraştığınızda bir evin metrekaresi için aykırı bir değer belirlemek gibi düşünebiliriz.)

##### Standart sapma

Veya verilerinizi gözden geçirdikten sonra standart sapmanızı kontrol eder ve ortalamadan belirli (örn. 2.5 standart sapma) standart sapma uzakta olan değerleri aykırı değerler olarak tanımlarsınız.

##### Z-Skoru

Z-puanı yönteminde ortalama 0 kabul edilir, ortalamadan bir standart sapma 1 z puanıdır. Bir eşik z-skor değeri belirlenir ve bu değer baz alınarak aykırı değerler hesaplanır.

##### Boxplot

![image.png](https://miro.medium.com/max/1400/1*i9b6wYv35jlr0DrcgZ3ZSQ.png)

Boxplot en çok tercih edilen yöntemdir. Hesapladığımız çeyrekler açıklığı (iqr) ile alt ve üst limitlerimizi belirliyoruz.

üst limit= q3+1.5x iqr

alt limit = q1–1.5 x iqr

IQR = q3 — q1

#### Kutu Grafiği Yönteminin Uygulanması

Gerekli import işlemlerini yaptıktan sonra veri ön işlemeye başlayabiliriz.

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
```

İlk olarak, çeyrekler açıklığı üzerinden üst ve alt limitleri belirleyelim.

```python
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]
```

Aykırı değer var mı?

```python
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None),

# "axis=None" olmasının nedeni, tüm verileri satır veya sütunlardan bağımsız olarak gözlemlemek istememizdir.
# Daha sonra bu süreci işlevselleştirebilir ve ileride kullanmak üzere kaydedebiliriz.
```

Daha sonra bu süreci fonksiyonlaştırabilir ve ileride kullanmak üzere kaydedebiliriz.

```python
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
outlier_thresholds(df, "Age")
```

Şimdiye kadar eşik değerleri belirledik, aykırı değerlere ulaştık ve aykırı değer olup olmadığını kontrol ettik.

Burada başka bir fonksiyon oluşturarak aykırı değer olup olmadığını hızlıca kontrol edebiliriz.

```python
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
```

Aykırı değerlere veya indeks bilgilerine ulaşmak istediğimizde aşağıdaki seçim işlemlerini kullanabiliriz.

```python
df[(df["Age"] < low) | (df["Age"] > up)].head()


df[(df["Age"] < low) | (df["Age"] > up)].index
```

Peki seçtiğimiz veri sayısal değilse bunu fonksiyona nasıl belirtebiliriz? Veya veri setindeki kategorik ve sayısal değişkenleri, hatta kategorik ve kategorik görünümlü sayısal değerleri sayısal bir görünümle belirten bir fonksiyonumuz olsaydı? Bunun için aşağıdaki adımları takip edebiliriz.

```python
def grab_col_names(dataframe, cat_th=10, car_th=20):

#Not: Sayısal görünüme sahip kategorik değişkenler de kategorik değişkenlere dahildir.

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car
```

Gördüğünüz gibi bunu bize her zaman işimize yarayabilecek kategorik, sayısal ve kategorik görünümlü ana sütunları gösteren bir fonksiyon olarak kullanabiliriz.
Aşağıdaki kod ile veri setimizdeki sütun isimlerini kaydedip görüntüleyebiliriz.

```python
cat_cols, num_cols, cat_but_car = grab_col_names(df)
```

Daha önce belirtildiği gibi, boxplot yöntemi yalnızca sayısal verilerde aykırı değerleri bulmamıza izin verdi. Artık “grab_col_names” fonksiyonu yardımıyla bulduğumuz sayısal verilerimizin aykırı değerlerini bulabiliriz.

```python
outlier_thresholds(df, num_cols)
```

#### Aykırı Değerler Sorununu Çözme

Aykırı değerler sorununun mevcut duruma göre farklı çözümleri vardır. Bu durumda aykırı değerlerden kurtulmak için onları silebilir, yok sayabilir veya eşikler değerler ile yeniden atayabiliriz.

##### Silme

```python
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
df["Age"] = df[~((df["Age"] < low) | (df["Age"] > up))]
#Yukarıdaki seçim işleminde kullandığımız “~” işareti sayesinde aykırı olmayan değerleri seçiyoruz.
```

##### Eşiklerle Değerler ile Yeniden Atama

Bazı senaryolarda, aykırı değerlere sahip diğer satır verileri önemli olduğunda, bunları silmek yerine baskılamamız gerekebilir. Bu gibi durumlarda, aykırı değerleri eşiklerle değiştiririz.

```python
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
df.loc[(df["Age"] > up), "Age"] = up_limit #upper treshold

df.loc[(df["Age"] < low), "Age"] = low_limit #lower treshold
```

##### Yoksaymak

Ancak ağaç yapılarını ele aldığımızda, modelimiz için bir sorun teşkil etmedikleri için aykırı değerleri görmezden geliyoruz. Özellikle ağaç modelleri model eğitme süresinde uzama oluşturabilmesi dışında aykırı değerlere karşı duyarsızdır.

Bu yazıda, veri ön işlemenin önemli bir parçası olan aykırı değerlerle başa çıkma konusunu uygulamalı bir şekilde incelemeye çalıştım. Umarım faydalı olmuştur.

Sormak istediğiniz sorular veya eklemek istediğiniz eleştirileriniz varsa yazmaktan çekinmeyin… 🤙🏻

#### Kaynaklar

“Box Plot Diagram to Identify Outliers.” n.d. Accessed April 14, 2022. https://www.whatissixsigma.net/box-plot-diagram-to-identify-outliers/.

“I Have An Outlier!” n.d. Accessed April 15, 2022. https://www.ctspedia.org/do/view/CTSpedia/OutLier.

“Identifying Outliers.” n.d. Accessed April 15, 2022. https://help.highbond.com/helpdocs/analytics/15/en-us/Content/analytics/analyzing_data/identifying_outliers.htm.

"If Your Data Is Bad Your Machine Learning Tools Are Useless." Accessed April 14, 2022. https://hbr.org/2018/04/if-your-data-is-bad-your-machine-learning-tools-are-useless

Resim 1.: https://lord-of-the-art.livejournal.com/461918.html
Resim 2.: https://storage.googleapis.com/publiclab-production/public/system/images/photos/000/032/980/original/Screen_Shot_2019-06-18_at_10.27.45_AM.png
