# #asdadsadsa
# type('mehaba')
# type(9)
# type(9.2)

# #string
# b = "asdasdas"
# b.islower()
# b.upper()

# a = "ASASASAS"
# a.isupper()
# a.lower()
# len(b)

# replace
# a = "eelelelelelelle"
# a.replace("e", "a")

# strip
# a = "   aaaaaaaa    "
# a.strip()
# a = "*****aaaaaaa*****"
# a.strip("*")

# method
# gel_yaz = "gelecegi_yazanlar"
# dir(gel_yaz)
# gel_yaz.capitalize()
# gel_yaz.title()

# # substrings
# gel_yaz = "gelecegi_yazanlar"
# # gel_yaz[1]
# # gel_yaz[0:3]
# gel_yaz[9:]

# type(1+2j)

# type converting
# birinci_sayi = input()
# ikinci_sayi = input()

# a =  float(birinci_sayi) + float(ikinci_sayi)
# b =  type(str(121212))

# lists
# nots = [50,60,70,80]
# a = ["a", 9.3, 9, nots]
# c = [a, nots]
# # print(a[0], a[3])
# print(c)
# # del c # deleting list

# list members
# liste = [10,20,30,40,50]
# print(liste[0:2])
# print(liste[2:])
# print(liste[2:-2])

# liste = ["a", 10, [20,30,40,50]]
# print(liste)
# print(liste[2])
# print(liste[1:2])
# print(liste[2][0])

# list members operations
# liste = ["a", "b", "c"]
# liste[0] = "yeni"
# print(liste)

# liste = ["a", "b", "c"]
# liste[0:3] = "yeni", "yeni", "yeni"
# added_list = liste + ["added", "added"]
# print(added_list)

# liste = ["a", "b", "c"]
# del liste[2]
# print(liste)

# list methods append, remove
# liste = ["a", "b", "c"]
# liste2 = ["d", "e", "f"]
# liste.append("abxc") #adding element to array
# print(liste)
# liste.remove("abxc") #remove element from array
# print(liste)

#insert, pop
# liste = ["a", "b", "c"]
# liste.insert(0, "added")
# liste.insert(2, "added2")
# liste.pop(4)
# liste.insert(len(liste), "son")
# print(liste)

# liste = ["a", "b", "c", "d", "e", "f", "f", "f"]
# print(liste.count("f"))
# print(liste.count("c"))
# liste_yedek = liste.copy()
# liste.clear()
# print(liste_yedek, liste)

# liste = ["a", "b", "c", "d"]
# liste.extend(["new", "new2", 10])
# print(liste)
# print(liste.index("c"))

# liste = ["a", "b", "c", "d", "e"]
# liste.reverse()
# print(liste)

# sort
# liste1 = ["b", "a", "d", "c"]
# liste2 = [1, 3, 0, 10, 7]
# liste1.sort()
# liste2.sort()
# print(liste1, liste2)

# clear
# liste = [ "a", "b", "c", "d"]
# liste.clear()
# print(liste)

# tuple data type (can not be changed)
# t = ("a","b","c",1,2,3,2.5, ["asd","gfdg"])
# t2 = "a", "b", "c"
# print(t)
# print(t2)

# if tuple has one elemen you have to put comma at the end
# t = ("eleman",)
# print(type(t))

# tuple elements
# t = ("ali", "veli", 1,2,3, [1,2,3,4])
# print(t[1])
# print(t[0:3])
# # t[2] = 99 can not be done

# dictionary has no "sıra"
# dic = {"REG": "Regression model", "LOJ": "Lojistic Regression", "CART":5}
# dic2 = {"a": ["b",10], "b": "asasd", "c":[1,2,3]}
# print(dic)
# print(len(dic))
# print(dic2)

# dic element operations
# dic = {"REG": "Regression model", "LOJ": "Lojistic Regression", "CART":5}
# dic2 = {"a": ["b",10], "b": "asasd", "c":[1,2,3]}
# dic3 = {"a": [{"b":"inside"},{"c":"inside2"}, 10], "b": "asasd", "c":[1,2,3]}
# print(dic["REG"], dic["LOJ"])
# print(dic2["a"])
# print(dic3["a"][0]["b"])
# print(dic3["a"][1]["c"])


# dic element add substruct
# dic = {"REG": "Regression model", "LOJ": "Lojistic Regression", "CART":5}
# dic["ABC"] = "asdasdasdasd"
# dic["REG"] = "aaaaaaaaaaaa"
# dic[1] = "111111"
# dic[2] = 111111
# inner_tuple = (1,2,3)
# dic[inner_tuple] = "yeni"
# #keys can not be changable
# print(dic)

# set (küme) no index - unique values - can be change
# l = ["a", "b", 100]
# s = set(l)
# print(s)

# t = ("a", "b", 100, 100)
# s = set(t)
# print(s)
# #deletes non-unique elements
# ali = "qwertyuıopğüasdfghjklşi,_zxcvbnm öç"
# s = set(ali)
# print(s)
# print(len(s))
# print(s[0]) # no index error unique elements


# set operations
# l = ["a", "b", "c", "c", "d"]
# s = set(l)
# s.add("ads")
# s.add("gip")
# s.remove("a")
# s.remove("b")
# #if element does not exist gives error on remove use discard
# s.discard("c")
# print(s)

# set difference(), intersect(), union(), symmetric_difference()
# s1 = set([1,2,5])
# s2 = set([1,2,3])
# print(s1.difference(s2))
# print(s2.difference(s1))

# print(s1.symmetric_difference(s2))

# print(s1.union(s2))
# print(s1.intersection(s2))

# converts s1 to intersection with s2
# s1.intersection_update(s2)
# print(s1)

# retrieve set
# s1 = set([7,8,9])
# s2 = set([5,6,7,8,9,10])

# print(s1.isdisjoint(s2))
# print(s1.issubset(s2))
# print(s2.issubset(s1))

# functions
# i = input("Sayı:")
# def square(a):
#    return a**2
# print(square(int(i)))

# i = input("Sayı:")
# def multiplyWithTwo(i):
#     print("Result: " + str(i*2)+ ",Girilen sayı: " + str(i))

# multiplyWithTwo(int(i))
# i1 = input("Sayı1:")
# i2 = input("Sayı2:")


# def multiply(a, b):
#     print(int(a)*int(b))

# multiply(i1, i2)

#default argument *************************************************************************
# i1 = input("Sayı1:")
# i2 = input("Sayı2:")

# def multiply(a = 1, b = 4):
#     print(int(a)*int(b))

# multiply()



#isi, nem, şarj

# def direk_hesap(isi, nem, sarj):
#     print((isi+nem)/sarj)

# direk_hesap(25,40,70)


# def direk_hesap(isi, nem, sarj):
#     return (isi+nem)/sarj

# cikti =  direk_hesap(25,25,70)

# print(cikti)


#Local Global variables
# x = 10 #Global
# y = 20 #Global

# def carp(x,y): #Local
#     return x*y #Local

# print(carp(2,3))


#Changing global variable from local area
# x = []

# def append(y):
#     x.append(y)
#     print(str(y) + "eklendi", x)

# append(1)
# append(2)


#control
# sinir=5000
# gelir=6000

# if (gelir < sinir):
#     print("küçük")
# elif (gelir == sinir): 
#     print("eşit")
# else:
#     print("büyük")


#Control with userInput
# sinir = 5000
# magaza_adi = input("Mağaza adı:")
# gelir = int(input("gelir:"))

# if gelir > sinir:
#     print("büyük")
# elif gelir < sinir:
#     print("küçük")
# else: print("eşit")


#Loops
# ogrenci = ["a","b","c","d","e"]

# for i in ogrenci:
#     print(i)

# maaslar = [100,200,300,400,500]

# def yeni_maas(maas):
#     if maas >= 400:
#         maas += maas*10/100
#     elif 200 < maas < 400:
#         maas += maas*15/100
#     elif maas <= 200:
#         maas += maas*30/100 
#     return maas

# for maas in maaslar:
#     print(yeni_maas(maas))

#continue break

# maaslar = [100,200,300,400,500]

#break
# for i in maaslar:
#     if i == 300:
#         print("kesildi")
#         break #finish loop permenently
#     print(i)


#continue
# maaslar = [100,200,300,400,500]

# for i in maaslar:
#     if i == 300:
#         continue #skips only one iteration
#     print(i)

#while
# sayi = 1

# while sayi < 9:
#     sayi +=1
#     print(sayi)


#object oriented programming

#class
# class VeriScientist():
#     print("asdasas")


#class variables
# class VeriScientist():
#     bolum = ""
#     sql="evet"
#     deneyim_yili = 0
#     bildigi_diller = []

# #class attributes
# # print(VeriScientist.sql)
# # print(VeriScientist.bolum)
# #alter class attributes
# # VeriScientist.sql ="hayir"
# # print(VeriScientist.sql)

# ali = VeriScientist()
# ali.bildigi_diller.append("Python")
# print(ali.sql,ali.deneyim_yili, ali.bolum, ali.bildigi_diller)

# veli = VeriScientist()
# print(veli.bildigi_diller)


#class
# class VeriBilimi():
#     bildigi_diller = ["R"]
#     sql="evet"
#     deneyim_yili = 0
#     bolum = ""
#     #sınıf örneklerine değişikllik yapılmasını sağlayan parça
#     def __init__(self):
#         self.bildigi_diller = []
#         self.bolum =  ""
#         self.deneyin_yili = 0
#         self.sql = 'evet'

# ali = VeriBilimi()
# ali.bildigi_diller.append('Python')
# ali.bolum = "asdasd"
# print(ali.bildigi_diller, ali.bolum)

# veli = VeriBilimi()
# veli.bildigi_diller.append('React')
# veli.bolum = "11ss"
# print(veli.bildigi_diller, veli.bolum)

# print(VeriBilimi.bildigi_diller, VeriBilimi.bolum)

#örnek methodları
# class DataScientist():
#     workers = []
#     def __init__(self):
#         self.bildigi_dersler = []
#         self.bolum = ''
#     def dil_ekle(self, yeni_dil):
#         self.bildigi_dersler.append(yeni_dil)

# ali = DataScientist()
# ali.dil_ekle('Python')
# print(ali.bildigi_dersler, ali.bolum)

# veli = DataScientist()
# veli.dil_ekle('R')
# print(veli.bildigi_dersler, veli.bolum)

#inheritence
# class Employees:
#     def __init__(self):
#         self.FistName=""
#         self.LastName=""
#         self.Address=""

# class DataScience(Employees):
#     def __init__(self):
#         self.Programming=""

# dataScientist1 = DataScience()
# dataScientist2 = DataScience()
# class Marketing(Employees):
#     def __init__(self):
#         self.StoryTelling=""

# mar1 = Marketing()
# mar2 = Marketing()


# class Employees:
#     def __init__(self, FirstName, LastName, Address):
#         self.FistName=FirstName
#         self.LastName=LastName
#         self.Address=Address

# employee = Employees('Aaaaa', 'Bbbbbb', 'asdasdasdasas')
# print(employee.FistName, employee.LastName, employee.Address)


#Functional Programming
#Pure Function

#E1: side effect
# A = 5

# def inpure_sum(b):
#     return b + A

# def pure_sum(a,b):
#     return a+b

# print(inpure_sum(6))
# print(pure_sum(6, 7))

#E2: Fatal Side Effects OOP
    # class LineCounter:
    #     def __init__(self, filename):
    #         self.file = open(filename, 'r')
    #         self.lines = []
    #     def read(self):
    #         self.lines = [line for line in self.file]
    #     def count(self):
    #         return len(self.lines)

    # lc = LineCounter('reading_data/deneme.txt')
    # print(lc.lines, lc.count())
    # lc.read()
    # lc.count()
    # print(lc.lines, lc.count())

#Functional
# def read(filename):
#     with open(filename, 'r') as f:
#         return [ line for line in f]

# def count(lines):
#     return len(lines)

# examples_lines = read('reading_data/deneme.txt')
# lines_count = count(examples_lines)
# print(lines_count, examples_lines)


# def old_sum(a,b):
#     return a+b

# old_sum(4,5)

# new_sum = lambda a, b: a+b
# print(new_sum(4,6))


# listExample = [('b',3), ('a',8), ('d',12), ('c',1)]

# #sort according to second attributes
# print(sorted(listExample, key = lambda x: x[1]))


#Vektorel Operations
#one dimension array is vector in analitic
#OOP
# a = [1,2,3,4]
# b = [2,3,4,5]

# ab = []

# for i in range(0, len(a)):
#     ab.append(a[i]*b[i])

# print(ab)
#FP
# import numpy as np

# a = np.array([1,2,3,4])
# b = np.array([2,3,4,5])

# print(a*b)


#map, filter, reduce
# listExample = [1,2,3,4,5]

# afterList = list(map(lambda x: x*10, listExample))
# print(afterList)

#filter
# listExample = [1,2,3,4,5,6,7,8,9,10]
# filteredList = list(filter(lambda x: x%2 == 0, listExample))
# print(filteredList)

#reduce
# from functools import reduce

# listExample = [1,2,3,4]
# reducedList = reduce(lambda a,b: a+b, listExample)
# print(reducedList)

#module (library, package)
# import checkModule as cm

# cm.yeni_maas(1000)

# from checkModule import yeni_maas
# yeni_maas(1000)

# import checkModule as cm

# print(cm.maaslar)


#exceptions
# a = 10
# b = 0

# try:
#     print(a/b)
# except ZeroDivisionError:
#     print("error")

# a = 10
# b = "2"

# try:
#     print(a/b)
# except TypeError:
#     print("asdasasd")
