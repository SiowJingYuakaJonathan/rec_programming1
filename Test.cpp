#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    int array[4];

    srand(time(0)); // Seed the random number generator with the current time

    for (int i = 0; i < 4; i++)
    {
        array[i] = 1000+ rand() % 9000;
    }
    
    

    cout << "Random 4-digit number: " << endl;
    for (int i = 0; i < 4; i++) {
        cout << array[i] << endl;
    }
   

    return 0;
}