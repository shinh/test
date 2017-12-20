class MyClass {
  MyClass() {
    x = 0;
    y = 1;
    mu_x = new Object();
    mu_y = new Object();
  }

  void addXToY() {
    // Because we update |x|...
    synchronized(mu_x) {
      x = getXPlusY();
    }
  }

  int getXPlusY() {
    // Because we read both |x| and |y|...
    synchronized(mu_y) {
      synchronized(mu_x) {
        return getX() + getY();
      }
    }
  }

  int getX() {
    synchronized(mu_x) {
      return x;
    }
  }

  int getY() {
    synchronized(mu_y) {
      return y;
    }
  }

  int x;
  // Protect |x|.
  Object mu_x;
  int y;
  // Protect |y|.
  Object mu_y;
}

public class recurrent_lock {
  public static void main(String[] args) throws Exception {
    MyClass obj = new MyClass();

    // Keep watching x+y.
    Thread thread = new Thread() {
        public void run() {
          for (int i = 0; i < 10000; i++) {
            System.out.println("x+y=" + obj.getXPlusY());
          }
        }
    };

    thread.start();
    for (int i = 0; i < 10000; i++) {
      obj.addXToY();
      Thread.sleep(1);
    }
  }
}

