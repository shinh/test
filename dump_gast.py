import ast
import gast
import sys

code = open(sys.argv[1]).read()
tree = ast.parse(code)
gtree = gast.ast_to_gast(tree)

for node in gtree.body:
    print(ast.dump(node))
