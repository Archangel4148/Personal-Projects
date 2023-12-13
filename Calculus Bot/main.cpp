#include "calcFunctions.h"

int main() {

  string function;
  string operators = "+-*/";
  vector<string> terms;

  cout << "Enter a function to differentiate: ";
  getline(cin, function);

  terms = splitString(function, ' ');

  cout << "Terms: " << terms << endl << endl;

  cout << "Derivative: " << differentiate(terms) << endl;

  return 0;
}