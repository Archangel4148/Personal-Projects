#ifndef PLAYING_CARDS
#define PLAYING_CARDS

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;

// Random number generator
static auto prng = [](){
        std::random_device rd;
        std::mt19937 gen(rd());
        return gen;
    }();

class Card {
    public:
        Card(string newName, string newSuit, int newValue);
        string name;
        string suit;
        int value;
        
        string cardInfo();
        
};

class Deck {

    private:
        vector<Card> cards;
        int cardCount = 52;

    public:
        Deck();
        Deck(vector<string> cardList);
        Card drawCard();
        void shuffle();
        void resetDeck();
        void emptyDeck();
        void addTop(Card topCard);
        Card getTop();
        int getSize();

        // == TESTING FUNCTION ==
        void printDeck();

};

double calculatePoints(vector<Card> hand);
Card findLowestCard(vector<Card> hand);
int getCardIndex(vector<Card> v, Card target);

string printCardVector(vector<Card> v);

#endif