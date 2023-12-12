#include "calcFunctions.h"

int main() {

  string function;
  vector<string> terms;

  cout << "Enter a function to differentiate: ";
  getline(cin, function);

  terms = splitString(function, ' ');

  cout << "Terms: " << terms << endl;

  return 0;
}