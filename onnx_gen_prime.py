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
    # Not sure why ONNX runtime needs this.
    range_tbl = gb.Reshape([range_tbl, gb.const([num])])
    return range_tbl


def calc_mod(gb, a, b):
    return gb.Sub([a, gb.Mul([gb.Div([a, b]), b])])


def gen_sieve_loop(initial_sieve_val, sieve_range_tbl):
    gb = onnx_script.GraphBuilder('gen_sieve')
    iter = gb.input('sieve_iter', 0)
    cond = gb.input('sieve_cond', True)
    sieve = gb.input('sieve_in', initial_sieve_val)

    n = gb.Add([iter, gb.const(2)])
    is_not_me = gb.Not([gb.Equal([sieve_range_tbl, n])])
    is_multiple = gb.Equal([calc_mod(gb, sieve_range_tbl, n), gb.const(0)])
    is_composite = gb.And([is_not_me, is_multiple])
    sieve = gb.Or([sieve, is_composite])

    gb.output(gb.const(True), True)
    gb.output(sieve, initial_sieve_val)
    return gb.make_graph()


def gen_primes_loop_then(initial_primes_val, primes, np):
    gb = onnx_script.GraphBuilder('primes_loop_then')
    gb.output(gb.Identity([primes]), initial_primes_val)
    gb.output(gb.Identity([np]), 0)
    return gb.make_graph()


def gen_primes_loop_else(initial_primes_val, primes, np, iter, range_tbl):
    gb = onnx_script.GraphBuilder('primes_loop_else')
    onehot = gb.Cast([gb.Equal([np, range_tbl])],
                     to=onnx.TensorProto.INT64)
    onehot = gb.Mul([onehot, gb.Add([iter, gb.const(2)])])
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
    then_graph = gen_primes_loop_then(initial_primes_val, primes, np)
    else_graph = gen_primes_loop_else(initial_primes_val, primes, np,
                                      iter, range_tbl)
    primes, np = gb.If([is_composite],
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


def get_primes_for_test(num_primes):
    sieve = get_sieve_for_test(num_primes * 10)
    np = 0
    i = 0
    primes = []
    while np < num_primes:
        if not sieve[i]:
            primes.append(i + 2)
            np += 1
        i += 1
    return primes


def gen_prime():
    max_val = 600
    # ONNX runtime crashes when this is not a multiple of 8.
    max_num_primes = 96
    #max_val = 1200
    #max_num_primes = 100
    gb = onnx_script.GraphBuilder('gen_prime')

    num_primes = gb.input('num_primes', max_num_primes)
    # ONNX runtime does not allow using inputs of enclosing graphs.
    num_primes = gb.Identity([num_primes])

    sieve_range_tbl = gen_range_tbl(gb, 'sieve', max_val)
    sieve_range_tbl = gb.Add([sieve_range_tbl, gb.const(2)])
    prime_range_tbl = gen_range_tbl(gb, 'prime', max_num_primes)

    initial_sieve_val = np.array([False] * max_val)

    sieve_loop = gen_sieve_loop(initial_sieve_val, sieve_range_tbl)
    sieve = gb.Loop([gb.const(max_val),
                     gb.const(True),
                     gb.const(initial_sieve_val)],
                    body=sieve_loop,
                    outputs=['sieve'])

    initial_primes_val = np.array([0] * max_num_primes)
    primes_loop = gen_primes_loop(initial_primes_val,
                                  sieve,
                                  prime_range_tbl,
                                  num_primes)
    primes, _ = gb.Loop([gb.const(max_val),
                         gb.const(True),
                         gb.const(initial_primes_val),
                         gb.const(0)],
                        body=primes_loop,
                        outputs=['primes', 'num_primes_out'])

    primes_val = get_primes_for_test(max_num_primes)
    gb.output(primes, primes_val)

    gb.gen_test(validate=True)


gen_prime()
