#include "playing_cards.h"

int main() {

  string response;
  bool winner = false;
  bool verbose;
  int rounds = 0;
  int wins = 0;
  int target;

  cout << "How many rounds would you like to simulate? "; // How many rounds to simulate before printing results
  cin >> target;
  cout << "Would you like a verbose output? (y/n): "; // Check if user wants everything to output information (slows WAAAAY down)
  cin >> response;

  if (response == "y" || response == "Y") { 
    verbose = true;
  }
  if (response == "n" || response == "N") {
    verbose = false;
  }

  // Begin actual simulation

  while (rounds < target) {

    Deck myDeck; // Main deck
    Deck discard(vector<string> {}); // Discard pile
    int players;
    vector<vector<Card>> playerHands; // List of each player's hand
    Card lowestCard = Card("BLANK", "NONE", 0);

    // Collect number of players (how many hands to fill)
    //cout << "Input the number of players: ";
    //cin >> players;
    //
    //while(players > 12 || players < 2) { // Input validation
    //  cout << "Sorry, this is an invalid number of players. Please enter a value from 2 to 12...\n\n";
    //  cout << "Input the number of players: ";
    //  cin >> players;    
    //}

    players = 3; // ===== TESTING VALUES =====

    // Initialize the discard pile for the start of the game
    discard.addTop(myDeck.drawCard());
    if (verbose) {
      cout << "Round " << rounds << endl;
      cout << "Top Card of Discard: " << discard.getTop().cardInfo()  << endl;
    }

    // Make a hand of three cards for each player
    for (int i = 0; i < players; ++i) {    
      vector<Card> thisHand;  
      playerHands.push_back(thisHand);
      if (verbose) {
        cout << "===============\nPlayer " << i + 1 << ": " << endl;
      }
      for (int j = 0; j < 3; ++j) {
        Card drawnCard = myDeck.drawCard(); // Draw three cards and add them to the hand(s)
        thisHand.push_back(drawnCard);
        if (verbose) {
          cout << "Added " << drawnCard.cardInfo() << endl;
        }
      }

      lowestCard = findLowestCard(thisHand);

      if (verbose) {
        cout << "Worst Card: " << findLowestCard(thisHand).cardInfo() << endl;
      }

      // Find the point value of each player's hand
      double points = calculatePoints(thisHand);
      if (verbose) {
        cout << "Points: " << points << endl << endl;
      }
      if (points == 31) {
        winner = true;
        wins++;
      }
    }
    rounds++;
  }
  
  //cout << setprecision(4) << "After " << rounds << " rounds, the calculated chance of being dealt a 31 is " << (static_cast<double>(wins)/static_cast<double>(rounds))*100 <<  "%, with " << wins << " wins out of " << rounds << " rounds.\n\n";
  
  return 0;
}