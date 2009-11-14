package main
func f(cc chan chan int) {
	c := make(chan int);
	cc <- c;
	i := <- c;
	println(i);
	cc <- c;
}
func main() {
	cc := make(chan chan int);
	go f(cc);
	c := <- cc;
	c <- 42;
	c <- 99;
	c = <- cc;
	//d := <- cc;
}
