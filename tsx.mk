ALL+=tsx
CLEAN+=$(ALL)

tsx: tsx.cc
	$(CXX) -O -g -std=c++11 -lpthread -fgnu-tm -march=corei7 -mtune=corei7 -mavx2 $< -o $@
