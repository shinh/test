import System
import List

fib = 1:1:zipWith (+) fib (tail fib)

main = do
  args <- getArgs
  print $ (0 *) $ (fib !!) $ read $ args !! 0
