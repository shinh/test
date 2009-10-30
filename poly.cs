interface Cmd {
    void run();
}

class Cmd_A : Cmd {
    public void run() {
        System.Console.Write("A");
    }
}

class Cmd_B : Cmd {
    public void run() {
        System.Console.Write("B");
    }
}

class poly {
    static void Main() {
        Cmd[] cmds = new Cmd[3];
        cmds[0] = new Cmd_A();
        cmds[1] = new Cmd_B();
        cmds[2] = new Cmd_A();
        int i;
        for (i = 0; i < 3; i++)
            cmds[i].run();
        System.Console.Write("\n");
    }
}
