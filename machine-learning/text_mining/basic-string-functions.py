# isim = "mvk"

# print(3*isim)
# print(isim + "mustafa")
# print("a" + isim[1:])

# names = ["ali", "veli", "ayse"]

# for i in names: 
#     print('Name:', i, sep = "")

# print("-----------------------")

# for i in names: 
#     print('_', i[0:], sep = "")

# print("-----------------------")

# print(*enumerate(names))

# print("-----------------------")

# for i in enumerate(names):
#     print(i)

# print("-----------------------")

# for i in enumerate("names"):
#     print(i)

# print("-----------------------")

# for i in enumerate(names, 1):
#     print(i)

# print("-----------------------")

# ##Array type Queries

# print("mvk".isalpha())
# print("mvk30".isalpha())
# print("123".isnumeric())
# print("123".isdigit())
# print("mvk30".isalnum())




# ## Elements and element indexes

# name = "mustafaakdogan"

# print(name[0:2])
# print(name.index("t"))
# print(name.index("a"))
# print(name.index("a", 5))



# ## Start and End Characters
# name = "mustafaakdogan"
# print(name.startswith("a"))
# print(name.startswith("m"))
# print(name.startswith("M"))
# print(name.endswith("a"))
# print(name.endswith("n"))




# ##count
# name = "mustafaakdogan"

# print(*sorted("defter"), sep="")




# ##Split
# name = "Mustafa Akdoğan"

# print(name.split())
# print(name.split("a"))




# ##Capital letters
# name = 'Mustafa Akdoğan'

# print(name.upper())
# print(name.lower())
# print(name.upper().lower())

# second_name = name.upper()
# print(second_name.islower())
# print(second_name.isupper())



# ##Capitalize

# name = "mustafa akdoğan"

# print(name.capitalize())
# print(name.title())

# name2 = "MUSTAFA akdoğan"

# print(name2.swapcase())



# ##Unwanted character strip lstrip rsting

# name = " hello "
# print(name.strip())
# name = "*hello*"
# print(name.strip("*"))
# name = "lhellol"
# print(name.strip("l"))
# name = "lhellol"
# print(name.lstrip("l"))
# name = "lhellol"
# print(name.rstrip("l"))




# ##Join
# name = "Mustafa Akdoğan Akdoğan"

# splitted = name.split()

# joiner = ""
# print(joiner.join(splitted))
# joiner = "*"
# print(joiner.join(splitted))




# ##Alter element replace, str.maketrans, translate
# name = "Mustafa Akdoğan Akdoğan"
# print(name.replace("A", "c"))
# print(name.replace("c", "A"))

# text = "Bu ifade İçerisinde bağzı TÜrkçe karakterler içermektedir"
# turkish_letters = "çÇğĞıİöÖşŞüÜ"
# fix_letters = "cCgGiIoOsSuU"

# fixed = str.maketrans(turkish_letters, fix_letters)
# print(text.translate(fixed))







# ##Contains
# import pandas as pd

# names = ["ayse", "Ayşe", "ali", "aali", "Ali", "veli", "mehmet", "berkcan"]
# v = pd.Series(names)
# print(v)
# print(v.str.contains("al"))
# print(v[v.str.contains("al")])
# print(v.str.contains("al").sum())
# print(v.str.contains("[aA]l").sum())