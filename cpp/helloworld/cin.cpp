#include <iostream>
#include <vector>

using namespace std;

int main() {

    int integerone; 
    int integertwo;

	// declare array and assign values
    cout << "Enter an integer between 1 and 100" << endl;
    cin >> integerone;

    cout << "Enter another integer between 1 and 100" << endl;
    cin >> integertwo;

    // output the difference
    cout << "The difference between your two numbers is: ";
    cout << integerone - integertwo << endl;


    return 0;
}