type 'a sep = Nil | Cons of ('a * 'a) sep

let rec size : 'a. 'a sep -> int = function
  | Nil -> 0
  | Cons x -> 2 * size x

;;

      
