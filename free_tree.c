#include <stdio.h>
#include <stdlib.h>

typedef struct tree {
    int val;
    struct tree* l;
    struct tree* r;
} tree;

tree* new_tree(int v) {
    tree* t = (tree*)malloc(sizeof(tree));
    t->val = v;
    t->l = NULL;
    t->r = NULL;
    return t;
}

void insert(tree* t, int v) {
    if (v < t->val) {
        if (t->l) {
            insert(t->l, v);
        }
        else {
            t->l = new_tree(v);
        }
    }
    else {
        if (t->r) {
            insert(t->r, v);
        }
        else {
            t->r = new_tree(v);
        }
    }
}

int free_cnt;
int free_checksum;
void wrap_free(tree* t) {
    free_cnt++;
    free_checksum += t->val;
    //printf("%d\n", t->val);
    free(t);
}

void free_tree(tree* t) {
    if (!t) return;
    free_tree(t->l);
    free_tree(t->r);
    wrap_free(t);
}

void stack_ni_yasasii_free_tree(tree* t) {
    tree done;
    tree* p = &done;
    while (t != &done) {
        while (t->l || t->r) {
            tree* n;
            if (t->l) {
                n = t->l;
                t->l = p;
            }
            else if (t->r) {
                n = t->r;
                t->r = p;
            }
            p = t;
            t = n;
        }

        wrap_free(t);
        t = p;
        if (t->l) {
            p = t->l;
            t->l = NULL;
        }
        else {
            p = t->r;
            t->r = NULL;
        }
    }
}

int main() {
    tree* r = new_tree(5);
    insert(r, 3);
    insert(r, 2);
    insert(r, 4);
    insert(r, 1);
    insert(r, 8);
    insert(r, 7);
    insert(r, 9);
    insert(r, 6);
    insert(r, 10);

    //free_tree(r);
    stack_ni_yasasii_free_tree(r);

    printf("%d %d\n", free_cnt, free_checksum);
}
