from phe import paillier

# 基于加法同态加密的点积协议
public_key, private_key = paillier.generate_paillier_keypair()
X = [1, 2, 3]
Y = [4, 5, 6]
EX = [public_key.encrypt(x) for x in X]
# EY = [public_key.encrypt(y) for y in Y]
# print(EX[0].ciphertext())
D = [EX[i].ciphertext() ** Y[i] for i in range(len(X))]

w = 1
for d in D:
    w = w * d

W_Enc = paillier.EncryptedNumber(public_key, w)
# w_p = w * public_key.encrypt(0)
# print(private_key.decrypt(w_p))

print(w)
