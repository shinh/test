#include <stdio.h>

#define PROTEIN "PROTEIN"
#define FAT "FAT"
#define CARBOHYDRATE "CARBOHYDRATE"
#define VITAMIN "VITAMIN"
#define MINERAL "MINERAL"

typedef char* nutrition;

typedef void (*support_fn)(int*);

typedef struct {
    support_fn support;
} Drink;

int human;

support_fn get_support_fn(const nutrition* nut) {
    static const nutrition* s_nut;
    s_nut = nut;
    void fn(int* human) {
        for (int i = 0; i < 5; ++i) {
            fprintf(stderr, "Got %s!\n", s_nut[i]);
        }
    }
    return fn;
}

#define Drink(x) {get_support_fn(&x)}

static const nutrition fiveMajorNutrients[5] = {
    PROTEIN,
    FAT,
    CARBOHYDRATE,
    VITAMIN,
    MINERAL,
};
int main() {
    Drink CalorieMateLIQUID = Drink(*fiveMajorNutrients);
    CalorieMateLIQUID.support(&human);
}
