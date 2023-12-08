#include "playing_cards.h"

int main() {

  Deck tempDeck;
  Deck mainDeck; // Main deck
  Deck discard(vector<string> {}); // Discard pile
  vector<int> scoreCounts(27, 0);
  vector<int> winScoreCounts(27, 0);
  string response;
  bool verbose = false;
  bool needToDraw = false;
  int players;
  int rounds = 0;
  int wins = 0, ties = 0;
  int target;
  double pointSum = 0;

  cout << "How many rounds would you like to simulate? "; // How many rounds to simulate before printing results
  cin >> target;
  if (target <= 20) {
    cout << "Would you like a verbose output? (y/n): "; // Check if user wants everything to output information (slows WAAAAY down)
    cin >> response;
  }

  if (response == "y" || response == "Y") { 
    verbose = true;
  }
  if (response == "n" || response == "N") {
    verbose = false;
  }

  // Collect number of players (how many hands to fill)
  cout << "Input the number of players: ";
  cin >> players;
    
  while(players > 12 || players < 2) { // Input validation
    cout << "Sorry, this is an invalid number of players. Please enter a value from 2 to 12...\n\n";
    cout << "Input the number of players: ";
    cin >> players;    
  }

  // Begin actual simulation

  while (rounds < target) {

    double maxPoints = 0;
    double startPoints = 0;
    int winningPlayer = 0;
    bool tied = false;
    vector<vector<Card>> playerHands; // List of each player's hand
    Card lowestCard = Card("BLANK", "NONE", 0);

    mainDeck = tempDeck;
    mainDeck.shuffle();
    discard.emptyDeck();

    // Initialize the discard pile for the start of the game
    discard.addTop(mainDeck.drawCard());
    if (verbose) {
      cout << "\n-- Round " << rounds + 1 << " --" << endl;
    }

    // Make a hand of three cards for each player
    for (int i = 0; i < players; ++i) {  
      if (i != 0) {
        needToDraw = true;
      }  
      vector<Card> thisHand;  
      playerHands.push_back(thisHand);
      if (verbose) {
        cout << "\n===============\nTop Card of Discard: " << discard.getTop().cardInfo()  << "\nPlayer " << i + 1 << ": " << endl;
      }
      for (int j = 0; j < 3; ++j) {
        Card drawnCard = mainDeck.drawCard(); // Draw three cards and add them to the hand(s)
        thisHand.push_back(drawnCard);
        if (verbose) {
          cout << "Added " << drawnCard.cardInfo() << endl;
        }
      }

      lowestCard = findLowestCard(thisHand);

      vector<Card> tempHand = thisHand;
      double points = calculatePoints(thisHand);
      int worstIndex = 0;

      if (needToDraw) {

        tempHand.push_back(discard.getTop());

        if (calculatePoints(tempHand) > calculatePoints(thisHand)) { // If the top discard is better than the original worst card:
        
          thisHand.push_back(discard.getTop()); // Draw from the discard
          if (verbose)
            cout << "Took the " << discard.getTop().cardInfo() << " from the discard pile\n";
        }
        else {
          Card drawnCard = mainDeck.drawCard();
          thisHand.push_back(drawnCard); // Draw from the deck
          if (verbose)
            cout << "Drew the " << drawnCard.cardInfo() << endl;
        }
        lowestCard = findLowestCard(thisHand); // Now recalculate the best card to discard based on the drawn card
        worstIndex = getCardIndex(thisHand, lowestCard);

        if (verbose)
          cout << "Worst Card: " << lowestCard.cardInfo() << endl;

        discard.addTop(thisHand[worstIndex]);
        thisHand.erase(thisHand.begin() + worstIndex);
        

        if (verbose)
          cout << "Discarded the " << discard.getTop().cardInfo() << endl;
        
        points = calculatePoints(thisHand);

        if (points > maxPoints) {
          maxPoints = points;
          winningPlayer = i + 1;
          tied = false;
        }
        else if (points == maxPoints) {
          tied = true;
        }

        if (i == 0) {
          startPoints = points;
          pointSum += points;
          if (points != 30.5)
            scoreCounts[points-6]++;
          else
            scoreCounts[26]++;
        }
      }
      if (verbose) {
        cout << "Points: " << points << endl << "===============" << endl;
      }
    }
    if (verbose) {
      if (!tied) {
        if (winningPlayer != 1)
          cout << "Winner: Player " << winningPlayer << endl;
      
        if (winningPlayer == 1)
          cout << "Winner: Player 1 (" << startPoints << ")" << endl;
      }
      else
        cout << "Winner: TIED!" << endl;
    }

    if (winningPlayer == 1 && !tied) {
      wins++;
      if (startPoints != 30.5)
        winScoreCounts[startPoints-6]++;
      else
        winScoreCounts[26]++;
    }
    if (tied)
      ties++;
    
    rounds++;
  }

  cout << fixed << setprecision(3) << "\n\nAfter " << target << " rounds, player one won " << static_cast<float>(wins)/static_cast<float>(target)*100 << "% (" << wins << "/" << target << ") of the rounds. (" << ties << " ties)\n";

  cout << setprecision(2) << fixed << "Average points for player one: " << pointSum / static_cast<float>(target) << endl;

  cout << "Points | Wins/Occurrences\n";
  for (int l = 0; l < 26; l++) {
    cout << "     " << l+6 << " : " << winScoreCounts[l] << "/" << scoreCounts[l] << fixed << setprecision(2) << " (" << static_cast<float>(winScoreCounts[l])/static_cast<float>(scoreCounts[l])*100 << "%)" << endl;
  }
  cout << "     30.5 : " << winScoreCounts[26] << "/" << scoreCounts[26] << fixed << setprecision(2) << " (" << static_cast<float>(winScoreCounts[26])/static_cast<float>(scoreCounts[26])*100 << "%)" << endl;


  cout << "\n\nPress Enter to exit...";
  cin.get();

  return 0;
}