struct S {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
}

fn new_s() -> S {
    S { a: 42, b: 99, c: 100, d: 0, e: 0, f: 0, g: 0 }
}

fn main() {
    let s = &new_s();
    println!("{} {} {} {} {} {} {}", s.a, s.b, s.c, s.d, s.e, s.f, s.g);
}
