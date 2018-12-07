import numpy as np
import onnx
import onnx_script


# For some reason, ONNX runtime does not like Greater for INT64.
def to_float(gb, v):
    return gb.Cast([v], to=onnx.TensorProto.FLOAT)


def gen_range_loop():
    gb = onnx_script.GraphBuilder('range_loop')
    iter = gb.input('range_iter', 0)
    cond = gb.input('range_cond', True)
    # To workaround ONNX's restriction for Loop. The number of inputs
    # must be greater than 2.
    dummy = gb.input('range_dummy', True)
    gb.output(gb.const(True), True)
    gb.output(gb.Identity([dummy]), True)
    gb.output(gb.Identity([iter]), 0)
    return gb.make_graph()


def gen_range_tbl(gb, name, num):
    _, range_tbl = gb.Loop([gb.const(num), gb.const(True), gb.const(True)],
                           body=gen_range_loop(),
                           outputs=[name + '_range_dummy_output',
                                    name + '_range_tbl'])
    # Not sure why we need this.
    range_tbl = gb.Reshape([range_tbl, gb.const([num])])
    return range_tbl


def gen_composites_loop():
    gb = onnx_script.GraphBuilder('gen_composites')
    iter = gb.input('composites_iter', 0)
    cond = gb.input('composites_cond', True)
    n = gb.input('n', 0)

    i = gb.Add([iter, gb.const(2)])
    is_greater = gb.Greater([to_float(gb, i), to_float(gb, n)])
    mod = gb.Mul([gb.Div([i, n]), n])
    is_multiple = gb.Equal([i, mod])
    is_composite = gb.And([is_greater, is_multiple])

    gb.output(gb.const(True), True)
    gb.output(gb.Identity([n]), 0)
    gb.output(is_composite, True)
    return gb.make_graph()


def gen_sieve_loop(initial_sieve_val):
    gb = onnx_script.GraphBuilder('gen_sieve')
    iter = gb.input('sieve_iter', 0)
    cond = gb.input('sieve_cond', True)
    sieve = gb.input('sieve_in', initial_sieve_val)

    n = gb.Add([iter, gb.const(2)])
    composites_loop = gen_composites_loop()
    _, composites_table = gb.Loop([gb.const(len(initial_sieve_val)),
                                   gb.const(True),
                                   n],
                                  body=composites_loop,
                                  outputs=['dummy', 'composites_table'])
    # Not sure why we need this.
    composites_table = gb.Reshape([composites_table,
                                   gb.const([len(initial_sieve_val)])])
    sieve = gb.Or([sieve, composites_table])

    gb.output(gb.const(True), True)
    gb.output(sieve, initial_sieve_val)
    return gb.make_graph()


def gen_primes_loop_then(initial_primes_val, primes, np):
    gb = onnx_script.GraphBuilder('primes_loop_then')
    gb.output(gb.Identity([primes]), initial_primes_val)
    gb.output(gb.Identity([np]), 0)
    return gb.make_graph()


def gen_primes_loop_else(initial_primes_val, primes, np, val, range_tbl):
    gb = onnx_script.GraphBuilder('primes_loop_else')
    onehot = gb.Cast([gb.Equal([np, range_tbl])],
                     to=onnx.TensorProto.INT64)
    onehot = gb.Mul([onehot, val])
    primes = gb.Add([primes, onehot])
    gb.output(primes, initial_primes_val)
    gb.output(gb.Add([np, gb.const(1)]), 0)
    return gb.make_graph()


def gen_primes_loop(initial_primes_val, sieve, range_tbl, num_primes):
    gb = onnx_script.GraphBuilder('primes_loop')
    iter = gb.input('prime_iter', 0)
    cond = gb.input('prime_cond', True)
    primes = gb.input('prime_primes', initial_primes_val)
    np = gb.input('np', 0)

    is_composite = gb.Gather([sieve, iter])
    val = gb.Add([iter, gb.const(2)])
    is_4949 = gb.const(False)
    ten = gb.const(10)
    for i in range(4):
        v = gb.Div([val, gb.const(10 ** i)])
        m = gb.Sub([v, gb.Mul([gb.Div([v, ten]), ten])])
        is_4949 = gb.Or([is_4949, gb.Equal([m, gb.const(4)])])
        is_4949 = gb.Or([is_4949, gb.Equal([m, gb.const(9)])])

    is_not_4949_prime = gb.Or([gb.Not([is_4949]), is_composite])

    then_graph = gen_primes_loop_then(initial_primes_val, primes, np)
    else_graph = gen_primes_loop_else(initial_primes_val, primes, np,
                                      val, range_tbl)
    primes, np = gb.If([is_not_4949_prime],
                       then_branch=then_graph,
                       else_branch=else_graph,
                       outputs=['next_primes', 'next_np'])

    cond = gb.Greater([to_float(gb, num_primes), to_float(gb, np)])
    gb.output(cond, True)
    gb.output(primes, initial_primes_val)
    gb.output(np, 0)
    return gb.make_graph()


def get_sieve_for_test(max_val):
    sieve = [False] * max_val
    for n in range(2, max_val + 2):
        for i in range(2, max_val):
            v = n * i - 2
            if v >= max_val: break
            sieve[v] = True
    return sieve


def get_4949_primes_for_test(num_primes):
    sieve = get_sieve_for_test(num_primes * 20)
    np = 0
    i = 0
    primes = []
    while np < num_primes:
        s = str(i + 2)
        if not sieve[i] and ('4' in s or '9' in s):
            primes.append(i + 2)
            np += 1
        i += 1
    return primes


def gen_prime():
    max_val = 1500
    # ONNX runtime crashes when this is not a multiple of 8.
    max_num_primes = 104
    gb = onnx_script.GraphBuilder('gen_4949_prime')

    num_primes = gb.input('num_primes', max_num_primes)
    # ONNX runtime does not allow using inputs of enclosing graphs.
    num_primes = gb.Identity([num_primes])

    range_tbl = gen_range_tbl(gb, 'prime', max_num_primes)

    initial_sieve_val = np.array([False] * max_val)

    sieve_loop = gen_sieve_loop(initial_sieve_val)
    sieve = gb.Loop([gb.const(max_val),
                     gb.const(True),
                     gb.const(initial_sieve_val)],
                    body=sieve_loop,
                    outputs=['sieve'])

    initial_primes_val = np.array([0] * max_num_primes)
    primes_loop = gen_primes_loop(initial_primes_val,
                                  sieve,
                                  range_tbl,
                                  num_primes)
    primes, _ = gb.Loop([gb.const(max_val),
                         gb.const(True),
                         gb.const(initial_primes_val),
                         gb.const(0)],
                        body=primes_loop,
                        outputs=['primes', 'num_primes_out'])

    primes_val = get_4949_primes_for_test(max_num_primes)
    gb.output(primes, primes_val)

    gb.gen_test(validate=True)


gen_prime()
