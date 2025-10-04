from func import Ket, Bra

a = Ket([1,0,3])
c = a * 2
print(c.coef)
d = a.dagger()
print(d)

b = Bra([3,4,6])
c = Ket([2,3,4])
print(b.tensor(b))
