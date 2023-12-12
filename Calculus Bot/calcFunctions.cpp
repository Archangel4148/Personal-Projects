#include "calcFunctions.h"

// ---- FUNCTION IMPLEMENTATION ----

vector<string> splitString(const string input, char delim) {
  vector<string> terms;
  string tempTerm = "";
  char character;

  for (int i = 0; i < input.length(); i++) {
    character = input[i];
    if (character != delim) {
      tempTerm += character;
    }
    else {
      terms.push_back(tempTerm);
      tempTerm = "";
    }

    if (i == input.length() - 1) {
      terms.push_back(tempTerm);
    }

  }

  return terms;
}


string differentiate(string term) {
  
  for (int i = 0; i < term.length(); i++) {
    cout << i << " : " << term[i] << endl;
  }

  if (term[term.length()-1] == 'x') {
    return string(term.substr(0, term.length() - 1));
  }

  return "ERROR";
}