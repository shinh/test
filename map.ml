let f n =
 let rec g i =
   if i>0 then g (i-1)*2 else n
  in
  List.map g [3; 4; n]
in
Printf.printf "%d\n" (List.nth (f 3) 2)
