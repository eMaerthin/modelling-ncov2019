#include "mocosMath.h"
#include <random>
#include "btrd.h"

std::random_device random_dev;
std::mt19937 random_gen(random_dev());
std::uniform_real_distribution<double> stdunif(0.0, 1.0);

void mocos_seed(std::mt19937::result_type seed)
{
    random_gen.seed(seed);
}
