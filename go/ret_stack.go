package main
import."fmt"
func f() *int {
	i := 42;
	return &i;
}
func main() {
	for i:=0; i<10000000;i++{
		Printf("%p\n",f());
	}
}
