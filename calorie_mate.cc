#include <iostream>
#include <string>
#include <vector>

typedef std::string nutrition;

nutrition PROTEIN = "PROTEIN";
nutrition FAT = "FAT";
nutrition CARBOHYDRATE = "CARBOHYDRATE";
nutrition VITAMIN = "VITAMIN";
nutrition MINERAL = "MINERAL";

struct Drink {
    explicit Drink(const nutrition& nuts) {
        for (int i = 0; i < 5; ++i) {
            nuts_.push_back((&nuts)[i]);
        }
    }

    void support(int* human) {
        for (nutrition nut : nuts_) {
            std::cerr << "Got " << nut << "!" << std::endl;
        }
    }

    std::vector<nutrition> nuts_;
};

int human;

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
