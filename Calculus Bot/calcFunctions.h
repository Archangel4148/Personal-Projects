#ifndef CALCFUNCTIONS_H
#define CALCFUNCTIONS_H

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// ---- FUNCTION DEFINITIONS ----

// Splits a provided polynomial into a vector of its component terms
vector<string> splitString(const string input, char delim);


// Computes the derivative of the provided function
string differentiate(string term);


// Overloading the output operator to out cout for a vector
template <class T>
ostream& operator << (ostream& os, const vector<T>& v) {

    os << '[';

    for (int i = 0; i < v.size(); i++) {
        os << v[i];
        if (i < v.size()-1) {
            os << ", ";
        }
    }

    os << ']';

    return os;
}

#endif