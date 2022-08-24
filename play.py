import math
import struct
import time

def play(dsp, msec, *a):
    fs = []
    for v in a:
        fs.append(440 * (2**(1/12.0)) ** v)

    t = 0
    delta = 1 / 8000
    while t < msec / 1000:
        v = 0
        for f in fs:
            v += math.sin(f * t * 2 * math.pi) + 1

        v = int((v / 10) * 127)
        assert v >= 0
        assert v < 256
        #print(v)
        dsp.write(chr(v).encode())
        #dsp.write(struct.pack(">i", v))
        dsp.flush()

        t += delta


C = -6
D = -4
E = -2
F = -1
G = 1
A = 3
B = 5
O = 12

with open("/dev/dsp", "wb") as dsp:
    t = 200
    play(dsp, t, E + O)
    play(dsp, t, E - 1 + O)
    play(dsp, t, E + O)
    play(dsp, t, E - 1 + O)
    play(dsp, t, E + O)
    play(dsp, t, B)
    play(dsp, t, D + O)
    play(dsp, t, B)

    #play(dsp, 1000, 3)
    #play(dsp, 1000, 3, 5)
    #play(dsp, 1000, 3, 5, 7)
    #play(dsp, 1000, 3, 5, 7, 13)
    #play(dsp, 1000, 4, 6, 8)

    # for i in range(5):
    #     play(dsp, 200, 3 + i * 12 - 24)
    #     play(dsp, 200, 5 + i * 12 - 24)
    #     play(dsp, 200, 7 + i * 12 - 24)
    #     play(dsp, 200, 8 + i * 12 - 24)
    #     play(dsp, 200, 10 + i * 12 - 24)
    #     play(dsp, 200, 12 + i * 12 - 24)
    #     play(dsp, 200, 14 + i * 12 - 24)
    #     #play(dsp, 200, 15 + i * 12 - 24)

    #play(dsp, 1000, 3)
    #play(dsp, 1000, 5)
    #play(dsp, 1000, 7)
    #play(dsp, 1000, 3, 5, 7)
    #play(dsp, 1000, -1)
    #play(dsp, 1000, -5, -3, -1)
    #play(dsp, 10000, 0, )
