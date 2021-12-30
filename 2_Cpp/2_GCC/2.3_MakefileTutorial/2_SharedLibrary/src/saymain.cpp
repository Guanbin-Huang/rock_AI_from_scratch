#include "say.h"

int main()
{
    extern Say librarysay;
    Say localsay = Say((char*)"Local instance of Say");
    sayhello();
    librarysay.sayThis((char*)"howdy");
    librarysay.sayString();
    localsay.sayString();
    return 0;
}