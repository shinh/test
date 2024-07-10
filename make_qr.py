import sys

import qrcode

qr = qrcode.QRCode()
qr.add_data(sys.argv[1])
for r in qr.get_matrix():
    l = ""
    for c in r:
        l += "#" if c else "."
    print(l)


