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

string differentiateTerm(string term) {
  if (term.find('x') != std::string::npos) {

    string afterCoeff = term.substr(term.find('x'));
    string stringCoeff = "";
    string output = "";
    double exp;
    double coeff;

    for (int i = 0; i < term.find('x'); i++) {
      stringCoeff += term[i];
    }

    if (afterCoeff.find('^') != std::string::npos) {
      exp = stod(afterCoeff.substr(afterCoeff.find('^')+1));
    }
    else 
      exp = 1;

    coeff = stod(stringCoeff);

    coeff *= exp;
    exp -= 1;

    output += stripDecimal(coeff);

    if (exp != 0) {
      output += "x";
    }

    if (exp > 1) {
      output += "^";
      output += stripDecimal(exp);
    }
    else if (exp < 0) {
      output += "^(";
      output += stripDecimal(exp);
      output += ")";
    }
    return output;
  }
  else
    return "0";
}

string differentiate(vector<string> polynomial) {
  string output = "";
  string diffTerm;
  for (string item : polynomial) {
    if (item == "+" || item == "-" || item == "*" || item == "/") {
      output += " " + item + " ";
    }
    else {
      diffTerm = differentiateTerm(item);
      if (diffTerm != "0") {
        output += diffTerm;
      }
      else {
        output.erase(output.length()-1);
        output.erase(output.length()-1);
        output.erase(output.length()-1);
      }
    }
  }

  return output;
}

string stripDecimal(double decimal) {
  string s = to_string(decimal);
    while(s[s.length()-1] == '0') {
      s.erase(s.length()-1);
      if (s[s.length()-1] == '.') {
        s.erase(s.length()-1);
        break;
      }
    }
  return s;
}
