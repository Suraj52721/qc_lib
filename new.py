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
