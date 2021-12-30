
#include <iostream>
#include "average.h"

int main(int argc, char* argv[])
{
    Average avg;
    avg.insertValue(30.2);
    avg.insertValue(88.8);
    avg.insertValue(3.002);
    avg.insertValue(11.0);
    std::cout << "Average = " << avg.getAverage() << "\n";
    return(0);
}