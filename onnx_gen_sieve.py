import numpy as np
import onnx_script


def gen_composites_loop(n):
    gb = onnx_script.GraphBuilder('gen_sieve')
    iter = gb.input('composites_iter', 0)
    cond = gb.input('composites_cond', True)
    # To workaround ONNX's restriction for Loop. The number of inputs
    # must be greater than 2.
    dummy = gb.input('composites_dummy', True)

    i = gb.Add([iter, gb.const(2)])
    is_greater = gb.Greater([i, n])
    mod = gb.Mul([gb.Div([i, n]), n])
    is_multiple = gb.Equal([i, mod])
    is_composite = gb.Mul([is_greater, is_multiple])

    gb.output(gb.const(True), True)
    gb.output(gb.Identity([dummy]), True)
    gb.output(is_composite, True)
    return gb.make_graph()


def gen_sieve_loop(initial_sieve_val):
    gb = onnx_script.GraphBuilder('gen_sieve')
    iter = gb.input('sieve_iter', 0)
    cond = gb.input('sieve_cond', True)
    sieve = gb.input('sieve', initial_sieve_val)

    n = gb.Add([iter, gb.const(2)])
    composites_loop = gen_composites_loop(n)
    _, composites_table = gb.Loop([gb.const(len(initial_sieve_val)),
                                   gb.const(True),
                                   gb.const(True)],
                                  body=composites_loop,
                                  outputs=['dummy', 'composites_table'])
    sieve = gb.Add([sieve, composites_table])

    gb.output(gb.const(True), True)
    gb.output(sieve, initial_sieve_val)
    return gb.make_graph()


def get_sieve_for_test(max_val):
    sieve = [False] * max_val
    for n in range(2, max_val + 2):
        for i in range(2, max_val):
            v = n * i - 2
            if v >= max_val: break
            sieve[v] = True
    return sieve


def gen_prime():
    max_val = 1200
    gb = onnx_script.GraphBuilder('gen_prime')

    num_prime = gb.input('num_prime', 100)

    initial_sieve_val = np.array([False] * max_val)

    sieve_loop = gen_sieve_loop(initial_sieve_val)
    sieve = gb.Loop([gb.const(max_val), gb.const(True),
                     gb.const(initial_sieve_val)],
                    body=sieve_loop,
                    outputs=['sieve'])

    gb.output(sieve, np.array(get_sieve_for_test(max_val)))

    gb.gen_test(validate=True)


gen_prime()
