import math # per importare le libreria si usa l'istruzione import
print(math.pi) # vediamo se funziona
print(math.cos(math.pi))
"""
print("Hello World")
print("Il nulla che \n nullifica")
print("Ora metto una sneaky \t tabulazione")
# La funzione print serve per stampare a schermo

#    Questo è un commento in
#    più linee

lato = float(input("Inserire la lunghezza del primo lato: "))
lato1 = float(input("Inserire la lunghezza del secondo lato: "))
print(lato * lato1)

# print("1" * "2")
print(1*2)

lato = input("Inserire la lunghezza del primo lato: ")
lato1 = input("Inserire la lunghezza del secondo lato: ")
# è sbagliato print(float(lato * lato1))
print(f"L'area è: {float(lato1) * float(lato)}")
"""
x = 5
print(f"Il valore di x: {x}")
x = x + 1
print(f"Il nuovo valore di x: {x}")
x = x**2
print(f"Il nuovissimo valore di x: {x}")
x = x * 2

a = [1, "ciao", "bella", 2.67, True, "1"]
print(a)
b = (5, 6, 7)
a[0] = "Cogito ergo sum"
print(a)
# linea fallace: b[1] = 2
a = [6, 8, 9]
print(a)

c = {
    "nome": "Francesco",
    "cognome": "Sermi",
    "eta": 20
}
print(c["eta"])
measure = [2, 8, 10, 12, 15, 17]
info_measure = {
    "length": len(measure),
    "sum": sum(measure),
    "average": sum(measure) / len(measure)
}
print(info_measure)
info_measure = {
    "length": (l := len(measure)),
    "sum": (s := sum(measure)),
    "avg": (s / l)
}
print(info_measure)
