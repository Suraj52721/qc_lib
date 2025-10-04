import numpy as np

class Ket:
    def __init__(self, coef):
        self.coef = np.array(coef, dtype=complex)

        # to represent real values as real vector
        t=0
        for a in coef:
            if np.imag(a) == 0:
                t=0
            else:
                t=1
                break
        if t==0:
            self.coef = np.array(coef, dtype=int)
    
    def __add__(self, other):
        if not isinstance(other, Ket):
            raise ValueError("Can only add another Ket.")
        return Ket(self.coef + other.coef)
    
    def __sub__(self, other):
        if not isinstance(other, Ket):
            raise ValueError("Can only subtract another Ket.")
        return Ket(self.coef - other.coef)

    def __mul__(self, scalar):
        return Ket(scalar * self.coef)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __repr__(self):
        return f'Ket({self.coef})'
    
    def dagger(self):
        return Bra(np.conjugate(self.coef))
    
    def inner_product(self, another):
        if not isinstance(another, Bra):
            raise ValueError('Inner Product can only be taken with Bra')
        else:
            return np.dot(self.coef, another.coef)
        
    def outer_product(self, another):
        if not isinstance(another, Bra):
            raise ValueError('Outer Product can only be taken with Bra')
        else:
            return np.outer(self.coef,another.coef)
        
    def tensor(self, another):
        if not isinstance(another, Ket):
            raise ValueError("Can Only Tensor product with Ket")
        else:
            m = []
            for a in self.coef:
               t = a * another.coef 
               m.extend(t)
            return Ket(m)


    
    
 
class Bra:
    def __init__(self,coef):
        self.coef = np.array(coef, dtype=complex).T

        # to represent real values as real vector
        t=0
        for a in coef:
            if np.imag(a) == 0:
                t=0
            else:
                t=1
                break
        if t==0:
            self.coef = np.array(coef, dtype=int)

    def __add__(self, other):
        if not isinstance(other, Bra):
            raise ValueError("Can only add another Bra.")
        return Bra(self.coef + other.coef)
    
    def __sub__(self, other):
        if not isinstance(other, Bra):
            raise ValueError("Can only subtract another Bra.")
        return Bra(self.coef - other.coef)

    def __mul__(self, scalar):
        return Ket(scalar * self.coef)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __repr__(self):
        return f'Bra({self.coef})'
    
    def dagger(self):
        return Ket(np.conjugate(self.coef))
    
    def inner_product(self, another):
        if not isinstance(another, Ket):
            raise ValueError('Inner Product can only be taken with Ket')
        else:
            return np.dot(self.coef, another.coef)
        
    def outer_product(self, another):
        if not isinstance(another, Ket):
            raise ValueError('Outer Product can only be taken with Ket')
        else:
            return np.outer(self.coef,another.coef)
        
    def tensor(self, another):
        if not isinstance(another, Bra):
            raise ValueError("Can Only Tensor product with Bra")
        else:
            m = []
            for a in self.coef:
               t = a * another.coef 
               m.extend(t)
            return Bra(m)


class Operator:
    def __init__(self, matrix):
        self.matrix = matrix



a = Ket([12,23,23])
print(a.coef)
b = Bra([12,34,45])
print(b.coef)

m = 23
c = m * a
print(c)
print(c.coef)



