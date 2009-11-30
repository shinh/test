package main

import "runtime"

func f(c chan int, i int) {
	//println(i);
	c <- i
}
func main() {
	runtime.GOMAXPROCS(4);

	c := make(chan int, 2);
	for i := 0; i < 100000; i++ {
		go f(c, i)
	}
	s := 0;
	for i := 0; i < 100000; i++ {
		j := <- c;
		//println(j);
		s += j;
	}
	println(s);
}
