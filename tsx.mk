ALL+=tsx
CLEAN+=$(ALL)

tsx: tsx.cc
	$(CXX) -O -g -std=c++11 -lpthread $< -o $@
