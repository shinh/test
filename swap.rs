fn swap(x: &mut String, y: &mut String) {
    let t = x.clone();
    *x = y.clone();
    *y = t;
}

fn main() {
    let mut x = "foo".to_string();
    let mut y = "bar".to_string();
    swap(&mut x, &mut y);
    println!("{} {}", x, y);
}
