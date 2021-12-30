


#include <iostream>

void sayhello(void); 

class Say{
private:
    char* string;
public:
    Say(char* str) 
    {
        string = str;
    }
    void sayThis(const char* str)
    {
        std::cout << str << " from a static library\n";
    }
    void sayString(void);

};