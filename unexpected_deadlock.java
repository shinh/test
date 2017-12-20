class Parent {
  synchronized void parentFunc() {
    System.out.println("hey parent");
  }
}

class Child extends Parent {
  Child() {
    mu = new Object();
  }

  synchronized void childFunc() {
    synchronized(mu) {
      System.out.println("hey child");
    }
  }

  void childFunc2() {
    synchronized(mu) {
      parentFunc();
    }
  }

  Object mu;
}

public class unexpected_deadlock {
  public static void main(String[] args) {
    Child c = new Child();
    Thread thread = new Thread() {
        public void run() {
          for (int i = 0; i < 1000; i++)
            c.childFunc2();
        }
    };
    thread.start();
    for (int i = 0; i < 1000; i++)
      c.childFunc();
  }
}
