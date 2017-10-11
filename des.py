# Initial Permutation Matrix
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# Inverse Permutation Matrix
InvP = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

# Permutation made after each SBox substitution
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

# Initial permutation on key
PC_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

# Permutation applied after shifting key (i.e gets Ki+1)
PC_2 = [14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32]

# Expand matrix to obtain 48bit matrix
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# SBOX represented as a three dimentional matrix
# --> SBOX[block][row][column]
SBOX = [
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    ],
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    ],
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    ],
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    ],
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    ],
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    ],

    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
     ],
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ]
]

# Shift Matrix for each round of keys
SHIFT = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


def str_to_bitarray(s):
    # Converts string to a bit array.
    bitArr = list()
    for byte in s:
        bits = bin(byte)[2:] if isinstance(byte, int) else bin(ord(byte))[2:]
        while len(bits) < 8:
            bits = "0" + bits  # Add additional 0's as needed
        for bit in bits:
            bitArr.append(int(bit))
    return (bitArr)


def bitarray_to_str(bitArr):
    # Converts bit array to string
    result = ''
    for i in range(0, len(bitArr), 8):
        byte = bitArr[i:i + 8]
        s = ''.join([str(b) for b in byte])
        result = result + chr(int(s, 2))
    return result


class DES():
    def __init__(self):
        self.password = None
        self.plaintext = None
        self.keylist = list()

    def left_shift(self, a, b, round_num):
        # Shifts a list based on a round number
        num_shift = SHIFT[round_num]
        if num_shift == 1:
            # left shift first half
            lastBit = a[0]
            for i in range(0, len(a) - 1):
                a[i] = a[i + 1]
            a[len(a) - 1] = lastBit

            # left shift second half
            lastBit = b[0]
            for i in range(0, len(b) - 1):
                b[i] = b[i + 1]
            b[len(b) - 1] = lastBit

        elif num_shift == 2:
            # left shift first half
            lastBit = a[1]
            secondToLastBit = a[0]
            for i in range(0, len(a) - 2):
                a[i] = a[i + 2]
            a[len(a) - 2] = secondToLastBit
            a[len(a) - 1] = lastBit

            # left shift second half
            lastBit = b[1]
            secondToLastBit = b[0]
            for i in range(0, len(b) - 2):
                b[i] = b[i + 2]
            b[len(b) - 2] = secondToLastBit
            b[len(b) - 1] = lastBit
        return a + b

    def createKeys(self):
        # This functions creates the keys and stores them in keylist.
        # These keys should be generated using the password.
        if not self.keylist:
            # convert password to bit array
            keyBitArr = str_to_bitarray(self.password)
            # apply permuted choice 1
            permutedKey = self.permute(keyBitArr, PC_1)
            # split permuted key into two halves
            c = [permutedKey[:28]]
            d = [permutedKey[28:]]
            k = list()
            for i in range(1, 17):
                k.append(self.left_shift(c[i - 1], d[i - 1], i - 1))
                c.append(k[i - 1][:28])
                d.append(k[i - 1][28:])
            # apply PC_2 on k
            for i in range(0, len(k)):
                self.keylist.append([])
                self.keylist[i] = self.permute(k[i], PC_2)

    def XOR(self, a, b):
        # xor function - This function is complete
        return [i ^ j for i, j in zip(a, b)]

    def performRound(self, left, right, roundNum):
        # Performs a single round of the DES algorithm
        left.append(right[roundNum])
        expendRight = self.permute(right[roundNum], E)
        xorBits = list()
        xorBits = self.XOR(expendRight, self.keylist[roundNum])
        sboxBit = list()
        sboxBit = self.sbox_substition(xorBits)
        finalPermutation = self.permute(sboxBit, P)
        finalPermutation = list(map(int, finalPermutation))
        right.append(self.XOR(left[roundNum], finalPermutation))

    def performRounds(self, text):
        # This function is used by the encrypt and decypt functions.
        # keys - A list of keys used in the rounds
        # text - The orginal text that is converted.
        left = list()
        right = list()
        left.append(text[:32])
        right.append(text[32:])
        for i in range(0, 16):
            self.performRound(left, right, i)
        return right[16] + left[16]

    def permute(self, bits, table):
        # Use table to permute the bits
        permutation = list()
        for bit in table:
            permutation.append(bits[bit - 1])
        return permutation

    def sbox_substition(self, bits):
        # Apply sbox subsitution on the bits
        sBits = list()
        for i in range(0, len(bits), 6):
            temp = [bits[i], bits[i + 1], bits[i + 2], bits[i + 3], bits[i + 4], bits[i + 5]]
            block = int(i / 6)
            row = int(str(temp[0]) + str(temp[5]), 2)
            column = int(str(temp[1]) + str(temp[2]) + str(temp[3]) + str(temp[4]), 2)
            s = SBOX[block][row][column]
            sBits += list(format(s, '04b'))
        return sBits

    def encrypt(self, key, plaintext):
        # Calls the performrounds function.
        self.password = key
        self.plaintext = plaintext
        self.createKeys()
        invpResult = list()
        textBitArr = str_to_bitarray(plaintext)
        for i in range(0, len(textBitArr), 64):
            currentBits = textBitArr[i:i+64]
            ipText = self.permute(currentBits, IP)
            result = self.performRounds(ipText)
            invpResult += self.permute(result, InvP)
        cipherText = bitarray_to_str(invpResult)
        return cipherText

    def decrypt(self, key, ciphertext):
        # Calls the performrounds function.
        self.password = key
        self.plaintext = ciphertext
        invpCipher = list()
        if not self.keylist:
            self.createKeys()
        self.reverse_key_list(self.keylist)
        cipherBitArr = str_to_bitarray(self.plaintext)
        for i in range(0, len(cipherBitArr), 64):
            currentBits = cipherBitArr[i:i+64]
            ipCipher = self.permute(currentBits, IP)
            roundsResult = self.performRounds(ipCipher)
            invpCipher += self.permute(roundsResult, InvP)
        plaintext = bitarray_to_str(invpCipher)
        return plaintext

    def reverse_key_list(self, keys):
        l = len(keys)
        for i in range(0, int(l/2)):
            temp = keys[l-1-i]
            keys[l-i-1] = keys[i]
            keys[i] = temp

if __name__ == '__main__':
    key = "9xc83vgs"
    plaintext = "a1b2c3d4e5f6g7h8"
    des = DES()
    ciphertext = des.encrypt(key, plaintext)
    text = des.decrypt(key, ciphertext)
    print(ciphertext)
    print(text)
