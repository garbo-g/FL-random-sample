"""
"""
import random, sys 
import math
import time
import numpy as np
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd 

rand = random_state(random.randrange(sys.maxsize))

class PrivateKey(object):
    '''Private Key = (\lambda, \mu) = (self.l, self.m)'''
    def __init__(self, p, q, n):
        if p == q:
            self.l = p * (p-1)   # self.l = \phi(n) = \lambda
        else:
            self.l = (p-1) * (q-1)   # self.l = \phi(n) = \lambda
        try:
            self.m = invert(self.l, n)   # m = \mu = (\phi(n))^{-1} mod n
        except ZeroDivisionError as e:
            print(e)
            exit()

class PublicKey(object):
    '''Public Key = (n, g) = (self.n, self.g)'''
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = mpz(rint_round(log2(self.n)))

def generate_prime(bits):    
    """Will generate an integer of b bits that is prime using the gmpy2 library  """    
    while True:
        possible =  mpz(2)**(bits-1) + mpz_urandomb(rand, bits-1 )
        if is_prime(possible):
            return possible

def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    #print(p)
    q = generate_prime(bits // 2)
    #print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)

def enc(pub, plain):#(public key, plaintext) #to do

    r = random.randint(0, sys.maxsize)
    while gcd(r, pub.n) != 1:
        r = random.randint(0, sys.maxsize)
    cipher = powmod(pub.g, plain, pub.n_sq) * powmod(r, pub.n, pub.n_sq) % pub.n_sq

    return cipher

def dec(priv, pub, cipher): #(private key, public key, cipher) #to do

    floor = (powmod(cipher, priv.l, pub.n_sq) - 1) // pub.n
    plain = floor * priv.m % pub.n

    return plain

def enc_add(pub, m1, m2): #to do
    """Add one encrypted integer to another"""

    m_added = (m1 * m2) % pub.n_sq

    return m_added


def enc_add_const(pub, m, c): #to do
    """Add constant n to an encrypted integer"""

    m_added_const = (m * powmod(pub.g, c, pub.n_sq)) % pub.n_sq

    return m_added_const


def enc_mul_const(pub, m, c): #to do
    """Multiplies an encrypted integer by a constant"""

    m_mul_const = powmod(m, c, pub.n_sq)

    return m_mul_const


if __name__ == '__main__':
    priv, pub = generate_keypair(20)
    
    '''
    Here we randomly generate 3 numbers: a,b and c, 
    where a and b are messages to be encrypted and c is the constant
    '''

    a = random.randint(0, math.floor(math.sqrt(pub.n)))
    b = random.randint(0, math.floor(math.sqrt(pub.n)))
    c = random.randint(0, math.floor(math.sqrt(pub.n)))

    a_enc = enc(pub, a)
    b_enc = enc(pub, b)

    a_dec = dec(priv, pub, a_enc)
    b_dec = dec(priv, pub, b_enc)

    print('Our Encryption and Decryption system is:', a == a_dec and b == b_dec)
    print('Our Encryption-Adding system is:', (a + b) % pub.n == dec(priv, pub, enc_add(pub, a_enc, b_enc)))
    print('Our Constant-Adding system is:', (a + c) % pub.n == dec(priv, pub, enc_add_const(pub, a_enc, c)))
    print('Our Constant-Multiplying system is:', (b * c) % pub.n == dec(priv, pub, enc_mul_const(pub, b_enc, c)))

        
    '''Test running time'''
    i = 0
    
    begin = np.zeros(100000)
    end = np.zeros(100000)
    OneShotTime = np.zeros(100000)


    while i < 100000:
        
        bits = random.randint(10, 1000)

        a = mpz_urandomb(rand, bits)
        b = mpz_urandomb(rand, bits)
    
        a_Enc = enc(pub, a)
        b_Enc = enc(pub, b)

        begin[i] = time.time()
        enc_add(pub, a_Enc, b_Enc)
        end[i] = time.time()

        print('Validation:', (a + b) % pub.n == dec(priv, pub, enc_add(pub, a_Enc, b_Enc)))

        OneShotTime[i] = end[i] - begin[i]

        i = i + 1

    AvarageTime = OneShotTime.sum() / 100000
        
    print("Avarage time of Paillier is:", AvarageTime)

