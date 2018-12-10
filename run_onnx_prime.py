import sys

import numpy as np
import onnxruntime


sess = onnxruntime.InferenceSession(sys.argv[1])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
primes = sess.run([label_name], {input_name: np.array(int(sys.argv[2]))})[0]
print(primes.shape)
print(primes)
