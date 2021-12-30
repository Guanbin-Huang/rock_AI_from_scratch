
#include "say.h"

void Say::sayString()
{
    std::cout << string << "\n";
    
}
Say librarysay((char*)"Library instance of Say");