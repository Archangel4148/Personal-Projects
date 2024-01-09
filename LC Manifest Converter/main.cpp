#include "converters.h"

int main() {

    string response;
    string line;
    vector<string> contents;
    bool validName = true;

    ofstream outFile;

    cout << "Welcome to the modpack manifest creator!\n\n";
            
    outFile.open("manifest.json");  
    outFile << "{\n";

    while (true) {

        validName = true;
        cout << "What is the name of your modpack? (a-z, A-Z, 0-9, _, NO SPACES!): ";
        
        cin >> response;
        if (isValidName(response)) {
            outFile << "\t\"name\": \"" << response << "\",\n";
            break;
        }   
    }

    while (!isValidVersion(response)) {
        cout << "Input the version number (for example, 1.0.1): ";
        cin >> response;
        if (!isValidVersion(response)) {
            cout << "Invalid input, please stick to the standard #.#.# ...\n\n";
        }
    }

    outFile << "\t\"version_number\": \"" << response << "\",\n";

    outFile << "\t\"website_url\": \"\",\n\t\"description\": \"";

    response = "";
    while (response.length() < 1 || response.length() > 250) {
        cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n'); 
        cout << "Input a description for your modpack (250 characters max): ";
        getline(cin, response);
        if (response.length() > 250) {
            cout << "Sorry, your description can only be up to 250 characters long (and must be at least 1 character)...\n";
        }
    }
    
    outFile << response << "\",\n";


    cout << "Paste your dependency array, then type \"done\": ";

    while (response != "done") {
        cin >> response;
        if (response != "done")
            contents.push_back(response);
    }

    outFile << "\t\"dependencies\": [\n";
    for (int i = 0; i < contents.size(); i++) {
        outFile << "\t\t\"" << contents[i] << "\"";
        if (i != contents.size() - 1) {
            outFile << ",";
        }
        outFile << endl;
    } 
    outFile << "\t]\n}";
}
