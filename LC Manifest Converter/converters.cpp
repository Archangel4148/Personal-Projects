#include "converters.h"

bool isValidName(string name) {
    for (char c : name) {
        if (!isalnum(c) && c != '_') {
            return false;
        }
    }
    return true;
}

bool isValidVersion(string version) {
    return (isdigit(version[0]) && version[1] == '.' && isdigit(version[2]) && version[3] == '.' && isdigit(version[4]) && version.length() == 5);
}
