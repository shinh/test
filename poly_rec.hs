data Sep a = Nil | Cons (Sep (a, a))

size :: Sep a -> Int
size Nil = 0
size (Cons x) = 2 * size x

     