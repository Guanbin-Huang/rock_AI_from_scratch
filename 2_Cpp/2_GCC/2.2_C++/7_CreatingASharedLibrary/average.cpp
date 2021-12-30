

#include "average.h"

void Average::insertValue(double value)
{
    count++;
    total += value;
}

int Average::getCount()
{
    return(count);
}

double Average::getTotal()
{
    return(total);
}

double Average::getAverage()
{
    return(total / (double)count);
}