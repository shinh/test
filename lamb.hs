import System
import Char

data T = Var Int 
       | Lam T
       | App T T
data V = Fun (V -> V)
       | Val Int
       | Str String

app :: V -> V -> V
app (Fun f) v = f v

eval :: T -> [V] -> V
eval (Var x) e = e !! x
eval (Lam t) e = Fun (\v -> eval t (v:e))
eval (App a b) e = eval a e `app` eval b e

unchurch :: V -> Int
unchurch c = i where 
	Val i = c `app` Fun inc `app` Val 0 
	inc (Val x) = Val (x+1)
church i = Fun (\f -> Fun (\x -> iterate (app f) x !! i))

-- \l . l (\a b i.unchurch a : unlist b) ""
unlist :: V -> String
unlist l = s where Str s = unlist' l
unlist' :: V -> V
unlist' l = l `app` Fun walk `app` Str "" where
	walk a = Fun (\b -> Fun (\i-> Str (chr (unchurch a) : unlist b)))

cons a b = Fun (\x -> x `app` a `app` b)
nil = Fun (\a -> Fun (\b -> b) )
tolist "" = nil
tolist (x:xs) = cons (church (ord x)) (tolist xs)

parse :: [Int] -> T
parse = fst . parse'
parse' :: [Int] -> (T, [Int])
parse' (0:0:s) = (Lam e, r) where (e, r) = parse'(s)
parse' (0:1:s) = (App e1 e2, r2) where
	(e1, r1) = parse'(s)
	(e2, r2) = parse'(r1)
parse' (1:s) = (Var (length a), tail b) where (a, b) = break (/=1) s

unpack :: String -> [Int]
unpack = concatMap (\x -> map (\p->(ord x) `div` 2^p `mod` 2) [7,6..0])

main=do
	[filename] <- getArgs
	source <- readFile filename
	interact (\i->unlist (eval (parse (unpack source)) [] `app` tolist i))