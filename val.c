int val = 42;
int mine;
void change_val() {
    val = 90;
    printf("%p %p\n", &val, &mine);
}

