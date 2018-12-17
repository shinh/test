import nnvm
import nnvm.symbol as sym
import nnvm.compiler.graph_util as graph_util

def sample():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z1 = sym.elemwise_add(x, sym.sqrt(y))
    z2 = sym.log(x)
    gradient = graph_util.gradients([z1, z2], [x, y])
    print(gradient)


#sample()


def nnvm_conv():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.conv2d(x, y, channels=3, kernel_size=3)
    grad = graph_util.gradients([z], [x, y])
    print(grad)


#nnvm_conv()


def nnvm_bn():
    x = sym.Variable("x")
    z = sym.batch_norm(x)
    grad = graph_util.gradients([z], [x])
    print(grad)


# fail
#nnvm_bn()


from tvm import relay

def relay_main():
    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.nn.conv2d(x, y, channels=3, kernel_size=(3,3))
    # ???
