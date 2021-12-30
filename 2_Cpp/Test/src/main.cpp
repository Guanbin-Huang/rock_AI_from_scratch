#include <memory>
#include <iostream>

using namespace std;

std::shared_ptr<int> bar(const int i)
{
    std::shared_ptr<int> sp(new int(i));
    return sp;
}

void foo(){
    auto sp = bar(1);
    cout << *sp;
}

int main()
{

    std::shared_ptr<int> sp (new int(1));
    auto sp2 = sp;
    auto sp3 = sp2;
    cout << sp3.use_count() << endl;
    cout << sp2.use_count() << endl;
    cout << sp.use_count() << endl;


    return 0;
}