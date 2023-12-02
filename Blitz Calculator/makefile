compiled: playing_cards.o main.o
	g++ -o compiled playing_cards.o main.o

%.o: %.cpp
	g++ -c $<

clean:
	rm -rf $(TARGET) *.o