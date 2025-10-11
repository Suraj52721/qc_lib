from func import Ket, Bra, Operator
import numpy as np
a = Ket([1,0,3])
c = a * 2
print(c.coef)
d = a.dagger()
print(d)

b = Bra([3,4,6])
c = Ket([2,3,4])
print(b.tensor(b))

d = Operator([[2,1,1],[1,2,1],[1,1,2]])
print(d)
print(d.unitary())
print(d.hermitian())
print(d.normal())

e = Operator([[2,1],[1,2]])
print(d.tensor(e).matrix)

f = Operator([[5,3+7j],[3-7j,2]])
print(f.spectral_decom())
print(d.op(c))


k = Bra([20,23,23])
h = Ket ([1+2j,2,0])

innerproduct = k.inner_product(h)
print(innerproduct)

t = Operator([[0,0,1],[3,4,5],[4,8,7]])
x = Operator([[0,0,3],[5,6,7],[2,3,4]])


q = Operator([[0,1],[1,0]])
spec = q.spectral_decom()


