#include "playing_cards.h"

Card::Card(string newName = "BLANK", string newSuit = "NONE", int newValue = 0) {
    name = newName;
    suit = newSuit;
    value = newValue;
}

string Card::cardInfo() {
    string info = name + " of " + suit;
    return info; 
}


Deck::Deck() {
    string suits[4] = {"clubs", "spades", "diamonds", "hearts"};
    string names[14] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"};
    int values[14] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 14; ++j) {
            Card newCard(names[j], suits[i], values[j]);
            cards.push_back(newCard);
        }
    }

    shuffle(); //Shuffles deck once it is initialized
    return;
}

Deck::Deck(vector<string> cardList) {
    cardCount = 0;
    for (int i = 0; i < cardList.size(); ++i) {
        int value = 0;
        string suit = "";
        string name = "";

        switch(cardList[i][0]) {
            case 'a':
                name = "Ace";
                value = 11;
                break;
            case 'k':
                name = "King";
                value = 10;
                break;
            case 'q':
                name = "Queen";
                value = 10;
                break;
            case 'j':
                name = "Jack";
                value = 10;
                break;
            case '0':
                name = "10";
                value = 10;
                break;
            default:
                name = cardList[i][0];
                value = static_cast<int>(cardList[i][0]);
                break;
        }
        switch(cardList[i][1]) {
            case 's':
                suit = "Spades";
                break;
            case 'c':
                suit = "Clubs";
                break;
            case 'd':
                suit = "Diamonds";
                break;
            case 'h':
                suit = "Hearts";
                break;
        }
        Card newCard = Card(name, suit, value);
        cards.push_back(newCard);
        cardCount++;
    }
    shuffle();
    return;
}



Card Deck::drawCard() {
    if (cardCount > 0) {  //If there are cards remaining, removes a random card from the deck, decreasing the deck size and returning the value.
        Card drawnCard = cards[0];
        for (int i = 0; i < cardCount - 1; i++) {
            cards[i] = cards[i+1];
        }
        cardCount--;
        return drawnCard;
    }
    else {
        Card errorCard("ERROR");
        return errorCard;
    }
}

void Deck::addTop(Card topCard) {
    cards.insert(cards.begin(), topCard);
    cardCount++;
    return;
}

Card Deck::getTop() {
    return cards[0];
}


void Deck::shuffle() { // Implementing Fisher-Yates shuffle algorithm
    srand(time(0));

    for(int i = cardCount - 1; i > 0; i--) {
        auto dist = std::uniform_int_distribution<int>{0, i};
        auto j = dist(prng);
        iter_swap(cards.begin() + i, cards.begin() + j);
    }
    return;
}

void Deck::resetDeck() {
    Deck newDeck;
    for (int i = 0; i < 52; i++) {
        cards[i] = newDeck.cards[i]; //Perform a deep copy of the template deck to reset the deck
    }
    cardCount = 52;
    shuffle();
    return;
}

void Deck::printDeck() {
    for (int i = 0; i < cardCount; i++) {
        cout << cards[i].name << " of " << cards[i].suit << endl;
        cout << endl;
    }
}

int Deck::getSize() {
    return cardCount;
}

double calculatePoints(vector<Card> hand) {
    vector<int> suitPoints = {0, 0, 0, 0};
    vector<string> suits = {"hearts", "diamonds", "spades", "clubs"};
    string potentialName = hand[0].name;
    bool matching = true;

    for (int i = 1; i < hand.size(); ++i) {
        if (hand[i].name != potentialName) {
            matching = false;
        }
    }
    if (matching) {
        return 30.5;
    }
    else {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < hand.size(); ++j) {
                if (hand[j].suit == suits[i]) {
                    suitPoints[i] += hand[j].value;
                }
            }
        }
        return static_cast<double>(*max_element(suitPoints.begin(), suitPoints.end()));
    }
}
