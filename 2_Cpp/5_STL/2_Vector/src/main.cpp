#include <iostream>
#include <list>
using namespace std;

void printList(const list<int>&l)
{
    for (list<int>::const_iterator it = l.begin(); it != l.end(); it++)
        cout << *it << " ";
    cout << endl;
}



int main()
{
    list<int> L1;
    L1.push_back(1);
    L1.push_back(7);
    L1.push_back(5);
    L1.push_back(10);

    printList(L1);

    L1.reverse();
    printList(L1);

    L1.sort();
    printList(L1);

    
    return 0;
}
















