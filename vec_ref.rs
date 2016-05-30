use std::vec::Vec;

fn f1() {
    let vec: Vec<i32> = vec!(42);
    let vref: &Vec<i32> = &vec;
    let i0: i32 = vref[0];
    let i1: i32 = (*vref)[0];
    let i2: i32 = (**vref)[0];
    assert_eq!(i0, i1);
    assert_eq!(i0, i2);
}

fn f2() {
    let vec: Vec<String> = vec!("foo".to_string());
    let vref: &Vec<String> = &vec;
    let s0: &str = &vref[0];
    let s1: &str = &*vref[0];
    let s2: &str = &(*vref)[0];
    let s3: &str = &(**vref)[0];
    let s4: &str = &*(*vref)[0];
    let s5: &str = &*(**vref)[0];
    let s6: &str = &*(&**vref)[0];
    let s7: &str = &&&&&&&*(&&&&&&&**vref)[0];
    let s8: &str = &*(&*&*&*&&***&&&&&*****&&&vref)[0];
    assert_eq!(s0, s1);
    assert_eq!(s0, s2);
    assert_eq!(s0, s3);
    assert_eq!(s0, s4);
    assert_eq!(s0, s5);
    assert_eq!(s0, s6);
    assert_eq!(s0, s7);
    assert_eq!(s0, s8);

    let a0: &[String] = &vref;
    let a1: &[String] = &*vref;
    let a2: &[String] = &**vref;
    assert_eq!(a0, a1);
    assert_eq!(a0, a2);
}

fn main() {
    f1();
    f2();
}
