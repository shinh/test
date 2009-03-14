class Base {}
trait Mixin {}
class Derived extends Base with Mixin {}

val ds: Array[Derived] = Array(new Derived, new Derived)
val objects: Array[_] = ds
val bs: Array[_ <: Base] = ds
val ms: Array[_ with Mixin] = ds
val bms: Array[_ >: Derived <: Object with Mixin] = ds

abstract class O {
  type I
}

val os: Array[x.I] forSome { val x: O }

val a: Array[Int] = Array(1, 2)
val b: Array[T] forSome { type T } = a
val c: Array[_] = a
val b: Array[T] forSome { type T } = a
