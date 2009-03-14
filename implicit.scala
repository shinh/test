object Implicit {
def main(args: Array[String]) {
class When(any: => Any) {
  def when(b: Boolean) = if (b) any
}
implicit def toWhen(any: => Any) = new When(any)

println("Hoge-") when true
println("Hoge-") when false

class C {
  def when(b: Boolean) {}
}
def myprintln(s: String): C = {
  println(s)
  new C
}

myprintln("Hoge-") when true
myprintln("Hoge-") when false

def myprintln3(s: String): Double = {
  println(s)
  1.1
}

myprintln3("Hoge-") when true
myprintln3("Hoge-") when false
}
}
