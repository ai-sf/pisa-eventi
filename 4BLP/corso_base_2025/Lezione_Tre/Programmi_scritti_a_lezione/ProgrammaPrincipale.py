import numpy as np
"""
a = np.array([[1, 0],[0, 1]])
c = np.array([[0, 1], [1, 0]])
print(a*c)
print(a @ c)

b = np.array([[1, 2], [3, 4], [5,6]])
d = np.array([[2, 4, 6], [5, 7, 9]])
print(b @ d)
print(b * d.T) # d.T

mat = b
print(f"mat: {mat}, id: {id(mat)}")
print(f"b: {b}, id: {id(b)}")
b[0][0] = 10
print(f"mat: {mat}, id: {id(mat)}")

mat = np.copy(b)
print(f"mat: {mat}, id: {id(mat)}")
b[0][0] = 20
print(f"mat: {mat}, id: {id(mat)}")

# Prima riga
print(f"Prima riga: {mat[0][:]}")
# Seconda colonna
print(f"Seconda colonna: {mat[:][1]}")

def area(a, b):
    prodotto = a * b
    return prodotto
    # return a*b

lato_1 = float(input("La dimensione della base: "))
lato_2 = float(input("La dimensione dell'altezza: "))
print(f"L'area è: {area(lato_1, lato_2)}")

if lato_1 >= 69:
    print("Nice!")
elif lato_1 >= 50:
    print("C'eravamo quasi...")
else:
    print("Sad :(")


###
if lato_1 >= 69 and lato_2 % 2==0:
    print("Condizione rispettata") 

if lato_1 > 30 or lato_2 % 2 == 1:
    print("L'or viene eseguito")
"""
import matplotlib.pyplot as plt
import numpy as np
xx = np.linspace(0, 2, 1000)
# Se xx = np.logspace(np.log(start), np.log(stop))
yy = xx ** 2
# plt.title("Grafico parabolozza")
plt.title(r"Grafico $x^2$")
plt.xlabel(r"$x \, \text{[m]}$")
plt.ylabel(r"$y \, [\text{m}^2]$")
plt.grid()
plt.plot(xx, yy)
plt.show()
lato_1 = float(input("Inserire lato_1: "))
lista = np.array([0, 1, 2, 3])
# lista = lista + 1
for x in range(0, len(lista)):
    print(x)
    lista[x] = lista[x] + 1
print(lista)

acc = 0
for x in lista:
    acc = acc + x

print(acc/len(lista))
print(lista)

while (lato_1 >= 69):
    lato_1 = float(input("Inserire lato_1: "))

