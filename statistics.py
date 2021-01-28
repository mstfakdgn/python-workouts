# import numpy as np

# popilation = np.random.randint(0, 80, 10000)
# # print(popilation[0:10])

# # #sample picking
# # #pick same sample everytime
# # np.random.seed(10)
# # sample = np.random.choice(a = popilation, size = 100)
# # print(sample[0:10])
# # print(sample.mean())
# print(popilation.mean())

# #sample distribution goes near to population mean  as we take more sample and avarage of them
# np.random.seed(10)
# sample1 = np.random.choice(a = popilation, size = 100)
# sample2 = np.random.choice(a = popilation, size = 100)
# sample3 = np.random.choice(a = popilation, size = 100)
# sample4 = np.random.choice(a = popilation, size = 100)
# sample5 = np.random.choice(a = popilation, size = 100)
# sample6 = np.random.choice(a = popilation, size = 100)
# sample7 = np.random.choice(a = popilation, size = 100)
# sample8 = np.random.choice(a = popilation, size = 100)
# sample9 = np.random.choice(a = popilation, size = 100)
# sample10 = np.random.choice(a = popilation, size = 100)
# sample11 = np.random.choice(a = popilation, size = 100)
# sample12 = np.random.choice(a = popilation, size = 100)

# samples = [ sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8, sample9, sample10,  sample11,  sample12]

# avarageMean = 0

# for i in samples:
#     avarageMean += i.mean()
# avarageMean = avarageMean / 12

# print(avarageMean)





















# import seaborn as sns 
# import researchpy as rp #install with pip get summary of attributes with tihs library

# tips = sns.load_dataset("tips")
# df = tips.copy()
# print(df.describe().T)

# #for numerical attributes
# print(rp.summary_cont(df[["total_bill", "tip", "size"]]))

# #for categorical attributes
# print(rp.summary_cat(df[["sex", "smoker", "day"]]))


# #varians beween to attributes
# print(df[["tip","total_bill"]].cov())

# #correlation
# print(df[["tip","total_bill"]].corr())


















# Confidal Range Theory
# confidal range covers every value that has samples has with given attribute
# how to calculate confidal range
# n= 100 , mean = 180, standart deviation = 40
# confidal range is 95 percent meaning that 95 percent of sample inside confidal range equals to z table value of 1,94-2,57
# mean +- s/squarerootof n = 180 +- 1,96 * 40/ squarerootof 100 = (172-188)this region is confidal arae
# only 5 percent of people outside of this area















# #Confidal Range Tutorial
# #using confidal range problem is
# # we have to give precise predicrtion on prcie of a product our prediction needs to be rlying on flexiblty and scientific truth
# # details
# # there is buyer, seller and product
# # buyers have been told that how much money they would pay for product that we are selling
# # we aim to find a price range that is flexible and accurate

# import numpy as np
# import statsmodels.stats.api as sms

# prices = np.random.randint(10,110, 1000)
# print(prices)
# print(prices.mean())

# # 95 percent of prices are inside output most appropiate price range
# print(sms.DescrStatsW(prices).tconfint_mean())















# #Possibilty
# # numerical presentation of a case happening

# # Rassal değişken => Takes its own value from an experiment output

# #bernolli posibilty distribution
# #successfull-unseccessfull, negative-positive, like these outputs are our subject this is discontinuous possibilty distribution 

# from scipy.stats import bernulli
# p =0.6
# rv = bernulli(p)
# print(rv.pmf(k=0))

















# # Big numbers law
# # As number gets bigger possibilty comes to normal
# import numpy as np

# rng = np.random.RandomState(123)

# for i in np.arange(1,21):
#     experiment_number = 2**i
#     head_tail = rng.randint(0,2, size = experiment_number)
#     head_possibility = np.mean(head_tail)
#     print('Number of flip:', experiment_number, "-----","Head Posibility: %.2f" % (head_possibility * 100))












#Binom distribution
# n tring k successful
# a coin is being flipped 4 times what is the possibilty of 2 tail


# For example a company is making advirtasiment throught some places so what is the posibilty of this advirtesiments click
# Advirtesiment is on distribution and clicking posibilty = 0.01 Question-> when 100 person saw the add what is the possibilty of clicking 1,5,10

# from scipy.stats import binom
# p = 0.01
# n = 100
# rv = binom(n, p)
# print(rv.pmf(1))
# print(rv.pmf(5))
# print(rv.pmf(10))


















# # Poisson Distribution
# # Used to calculate the probability of events that rarely occur in a given area in a given time period.
# # Example in a college what is the entering 5 wrong note out of 5000 student Lambda = 0.2
# #Calculating error For example there is a notice site people are evaluating the site for one year. We now it is poisson and Lambda = 0.1
# # which is avarage error number what is the probabilty of 0 error, 3 error and 5 error
# from scipy.stats import poisson

# lambda_ = 0.1 
# rv = poisson(mu = lambda_)
# print(rv.pmf(k = 0))
# print(rv.pmf(k = 3))
# print(rv.pmf(k = 5))













# #Normal Distribution
# #Calculating sold product prices we want to predict next months selling numbers
# # We know that distribution is normal Avarage sell number 80k, standar deviation 5k, what is the probabilty of more than 90k sell

# from scipy.stats import norm

# # 1 is all output is the moren tahn 90k
# print(1-norm.cdf(90,80,5))

# #more tahn 70k
# print(1-norm.cdf(70,80,5))
# #less than 73k
# print(norm.cdf(73,80,5))
# #between 85k and 90k
# print(norm.cdf(90,80,5) - norm.cdf(85,80,5))













#Hipotez test

#single sample T test
#between population mean and hypotatical is there a meaningfull difference


#application
#there is 5 steps while ordering product
#every step 20 second  we qeustion 4. step
#we get 100 sample
#standar deviation 5 sn, mean is 19sn

