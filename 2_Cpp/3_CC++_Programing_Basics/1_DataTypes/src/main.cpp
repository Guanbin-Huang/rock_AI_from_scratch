#include <iostream>
using namespace std;

int invertBit(int num)
{
   // 6x4 = 24, 4x2 = 8
   int res = (num & 0x000000FF) << 24 | \
             (num & 0x0000FF00) << 8  | \
             (num & 0x00FF0000) >> 8  | \
             (num & 0xFF000000) >> 24;
   return res;
}


int main(){

   
}

