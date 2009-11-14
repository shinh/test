package main

type Vector []float64

// Apply the operation to n elements of v starting at i.
func (v Vector) DoSome(i, n int, u Vector, c chan int) {
    for ; i < n; i++ {
        //v[i] += u.Op(v[i])
        v[i] += v[i]
    }
    c <- 1;    // signal that this piece is done
}

const NCPU = 4	// number of CPU cores

func (v Vector) DoAll(u Vector) {
    c := make(chan int, NCPU);  // Buffering optional but sensible.
    for i := 0; i < NCPU; i++ {
        go v.DoSome(i*len(v)/NCPU, (i+1)*len(v)/NCPU, u, c);
    }
    // Drain the channel.
    for i := 0; i < NCPU; i++ {
        <-c    // wait for one task to complete
    }
    // All done.
}

func main() {
	u := make(Vector, 1000000);
	v := make(Vector, 1000000);
	v.DoAll(u);
}
